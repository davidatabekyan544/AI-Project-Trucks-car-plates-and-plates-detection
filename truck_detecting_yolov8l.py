from ultralytics import YOLO

def train():
    model = YOLO("yolov8l.pt")

    model.train(
        data="C:/Users/davit/Downloads/vehicles.v2-release.yolov8/data.yaml",
        epochs=120,
        imgsz=640,
        batch=2,
        patience=25,
        device=0,
        optimizer="AdamW"
    )

if __name__ == "__main__":
    train()