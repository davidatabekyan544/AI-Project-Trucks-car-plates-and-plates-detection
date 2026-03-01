from ultralytics import YOLO

def main():

    model = YOLO(r"C:\Users\davit\PycharmProjects\AI-Project-Trucks-car-plates-and-plates-detection\runs\detect\train10\weights\best.pt")

    metrics=model.val(
        source=r"C:\Users\davit\Downloads\car_plates\images\test",
        save=True,
        batch=2,
        plots=True
    )

    print(metrics)



if __name__ == "__main__":
    main()