from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="/root/autodl-tmp/dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=128,
    dropout=0.3,
    hsv_h=0.02,
    hsv_s=0.8,
    conf=0.25,
    iou=0.7,
    max_det=100,
    patience = 20
)
