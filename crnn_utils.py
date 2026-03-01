import torch
import cv2
from arm_plates_dataset_loader import ALPHABET

CHARSET = ALPHABET


def preprocess_plate(plate_img):

    # Resize to model input size
    plate_img = cv2.resize(plate_img, (320, 64))

    # Sharpening (optional but safe)
    plate_img = cv2.GaussianBlur(plate_img, (3, 3), 0)

    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

    plate_img = plate_img / 255.0

    tensor = torch.tensor(plate_img).permute(2, 0, 1).unsqueeze(0)

    return tensor.float().cuda()


def decode_ctc_output(preds):

    preds = preds.softmax(2)
    preds = preds.argmax(2)

    charset = ALPHABET

    text = ""
    previous = -1

    for p in preds[0]:

        idx = p.item()

        if idx != previous and idx < len(charset):

            if charset[idx] != "<blank>":
                text += charset[idx]

        previous = idx

    return text


def recognize_text(model, image_tensor):

    with torch.no_grad():
        preds = model(image_tensor)
        text = decode_ctc_output(preds)

    return text