from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")

    model.train(
        data="C:/Users/davit/PycharmProjects/AI-Project-Trucks-car-plates-and-plates-detection/plates.yaml",
        epochs=40,
        imgsz=640,
        batch=8,
        device=0,
        optimizer="AdamW",
        patience=25
    )

if __name__ == "__main__":
    train()