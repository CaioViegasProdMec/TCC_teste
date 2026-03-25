import cv2                          # OpenCV - captura vídeo da webcam e processa imagens
import numpy as np                  # Arrays - Usado pelo OpenCV e YOLO
from ultralytics import YOLO        # Modelo de visão computacional que detecta objetos
import paho.mqtt.client as mqtt     # Cliente MQTT - envia mensagens para o broker
import json                         # formatação dos dados em json para o envio
import time                         # cooldown para detecções
from datetime import datetime       # Marcação do horário exato para cada detecção

## Ideias

# Percepcação de x y z
# Dar uma olhada no ligação com o arena para input de dados

# ============ CONFIGURAÇÕES ============
MQTT_BROKER = "localhost"           # Endereço do MQTT -> O localhost sigifica que ele está rodando na mesma máquina
MQTT_PORT = 1883                    # Porta padrão para o MQTT
MQTT_TOPIC = "fabrica/esteira/objeto"   # Ideia parecida com a assinatura Pub/Sub : O MQTT está enviando os dados para tópico e quem estiver assinando pode receber os dados

# Inicializa o modelo YOLO (usa modelo pré-treinado leve)

# Para objetos de uma fábrica simples, usado um modelo genérico
model = YOLO('yolov8n.pt')  # Modelo neural pré-treinado 'n' = nano (mais leve, roda em CPU), 80 tipos diferentes de objetos

# Dicionário para rastrear objetos já detectados (evita duplicatas)
objetos_detectados = set()                  # Conjunto vazio que não permite duplicatas
tempo_ultima_deteccao = {}
DETECTION_COOLDOWN = 4  # segundos

# ============ MQTT ============            # Client: qualquer dispositivo ou programa que se conecta ao Broker para enviar ou receber informações.

def conectar_mqtt():                    
    """Conecta ao broker MQTT"""
    client = mqtt.Client()                              # Criar um cliente do MQTT
    try:                                        
        client.connect(MQTT_BROKER, MQTT_PORT, 60)      # método que conecta ao broker, 60 segundos de tempo limite
        print(f" Conectado ao MQTT broker em {MQTT_BROKER}:{MQTT_PORT}")
        return client
    except Exception as e:
        print(f" Erro ao conectar MQTT: {e}")
        return None
    


def publicar_evento(client, objeto, confianca):
    """Publica a detecção no MQTT"""
    if client is None:                                   # Caso tenha dado algum erro para criar o client
        return
    
    evento = {
        "timestamp": datetime.now().isoformat(),
        "objeto": objeto,
        "confianca": round(float(confianca), 2),
        "id": f"{objeto}_{int(time.time())}"             # Retorna um timestamp Unix (segundos desde 1970)
    }
    
    client.publish(MQTT_TOPIC, json.dumps(evento))       # Envia uma mensagem para o MQTT, utilizando o MQTT_TOPIC (canal), transformando o dicionário evento em .json 
    print(f" Publicado: {evento}")




# ============ PROCESSAMENTO ============
def processar_frame(frame, client):
    """Processa um frame da webcam e publica detecções"""
    # Executa inferência YOLO
    results = model(frame, verbose=False)                 # frame = passa a imagem para a IA analisar ; verbose = False : não mostra as mensagens de progresso 
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # Pega classe e confiança
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                nome_classe = model.names[cls]
                
                # Filtra apenas objetos que interessam para a fábrica
                # Você pode ajustar esta lista conforme necessário
                objetos_uteis = ['person', 'bottle', 'cup', 'book', 'cell phone']
                
                if nome_classe in objetos_uteis:
                    # Controle para não publicar o mesmo objeto repetidamente
                    chave = f"{nome_classe}_{int(box.xyxy[0][0])}"
                    agora = time.time()
                    
                    if chave not in tempo_ultima_deteccao or \
                       (agora - tempo_ultima_deteccao[chave]) > DETECTION_COOLDOWN:
                        tempo_ultima_deteccao[chave] = agora
                        publicar_evento(client, nome_classe, conf)
                        
                        # Desenha no frame (feedback visual)
                        x1, y1, x2, y2 = box.xyxy[0]
                        cv2.rectangle(frame, (int(x1), int(y1)), 
                                     (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{nome_classe} {conf:.2f}", 
                                   (int(x1), int(y1)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    return frame

# ============ MAIN ============
def main():
    print(" Iniciando sistema de percepção do Gêmeo Digital")
    print(" Conectando à webcam...")
    
    # Conecta ao MQTT
    mqtt_client = conectar_mqtt()
    
    # Abre webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Não foi possível acessar a webcam")
        return
    
    print(" Webcam ativa. Pressione 'q' para sair.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Processa o frame
            frame_processado = processar_frame(frame, mqtt_client)
            
            # Mostra o resultado
            cv2.imshow('Percepção - Gêmeo Digital', frame_processado)
            
            # Sai ao pressionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n Interrompido pelo usuário")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if mqtt_client:
            mqtt_client.disconnect()
        print(" Sistema encerrado")

if __name__ == "__main__":
    main()