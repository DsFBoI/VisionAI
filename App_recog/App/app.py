from ultralytics import YOLO
import cv2

# Cargar el modelo personalizado entrenado
model = YOLO('C:/Users/danel/Downloads/cosas/VisionAI/runs/detect/train8/weights/best.pt')

# Configurar la captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar detección
    results = model(frame)

    # Visualizar resultados
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Obtener coordenadas del cuadro delimitador y confianza
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = box.cls[0]

            # Dibujar el cuadro delimitador y el texto en el frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Detección en tiempo real', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
