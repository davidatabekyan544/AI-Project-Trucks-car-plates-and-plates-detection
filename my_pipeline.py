from ultralytics import YOLO
import torch
import cv2
from crnn_utils import preprocess_plate, recognize_text
from utils import draw_label
from CRNN_model import CRNN
from arm_plates_dataset_loader import ALPHABET

torch.backends.cudnn.benchmark = True

# ===============================
# Load Models
# ===============================

truck_model = YOLO(r"C:\Users\davit\PycharmProjects\AI-Project-Trucks-car-plates-and-plates-detection\runs\detect\train2\weights\best.pt")

plate_model = YOLO(r"C:\Users\davit\PycharmProjects\AI-Project-Trucks-car-plates-and-plates-detection\runs\detect\train10\weights\best.pt")

truck_model.to("cuda")
plate_model.to("cuda")

# ===============================
# Load CRNN Model
# ===============================

num_classes = len(ALPHABET) + 1

crnn_model = CRNN(num_classes).to("cuda")

crnn_model.load_state_dict(torch.load(
    r"C:\Users\davit\PycharmProjects\AI-Project-Trucks-car-plates-and-plates-detection\arm_plate_crnn.pth",
    map_location="cuda"
))

crnn_model.eval()

# ===============================
# Pipeline Function
# ===============================

def process_frame(frame):

    # Test-time stabilization (do not retrain models)
    #frame = cv2.resize(frame, (1280, 720))

    # ---------- Truck Detection ----------
    truck_results = truck_model.predict(
        frame,
        imgsz=640,
        conf=0.18,
        augment=True,
        verbose=False
    )[0]

    for truck_box in truck_results.boxes.xyxy:

        x1, y1, x2, y2 = map(int, truck_box[:4])

        if x2 <= x1 or y2 <= y1:
            continue

        # Draw truck box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        truck_crop = frame[y1:y2, x1:x2]

        if truck_crop.shape[0] < 20 or truck_crop.shape[1] < 20:
            continue

        # ---------- Plate Detection ----------
        plate_results = plate_model.predict(
            truck_crop,
            imgsz=640,
            conf=0.2,
            augment=True,
            verbose=False
        )[0]

        h, w = truck_crop.shape[:2]

        for plate_box in plate_results.boxes.xyxy:

            px1, py1, px2, py2 = map(int, plate_box[:4])

            padding_x = int((px2 - px1) * 0.15)
            padding_y = int((py2 - py1) * 0.25)

            px1 = max(0, px1 - padding_x)
            py1 = max(0, py1 - padding_y)
            px2 = min(w, px2 + padding_x)
            py2 = min(h, py2 + padding_y)

            plate_crop = truck_crop[py1:py2, px1:px2]

            if plate_crop.shape[0] < 10 or plate_crop.shape[1] < 10:
                continue

            plate_tensor = preprocess_plate(plate_crop)

            text = recognize_text(crnn_model, plate_tensor)

            frame = draw_label(frame, text, (x1 + px1, y1 + py1))

    return frame