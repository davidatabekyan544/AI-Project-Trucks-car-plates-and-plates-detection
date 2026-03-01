from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\davit\PycharmProjects\AI-Project-Trucks-car-plates-and-plates-detection\runs\detect\train2\weights\best.pt")

    metrics = model.val(
        data="C:/Users/davit/Downloads/vehicles.v2-release.yolov8/data.yaml",
        split="test",
        batch=2
    )

    print(metrics)


if __name__ == "__main__":
    main()