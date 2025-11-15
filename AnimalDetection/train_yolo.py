from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=4,
    project='runs/animal_yolo',
    name='animal_detection',
    exist_ok=True
)
