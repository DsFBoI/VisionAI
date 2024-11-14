from ultralytics import YOLO


model =  YOLO("yolov8n.yaml")

result = model.train(data = "config.yaml", batch = 16, epochs =  100 )
