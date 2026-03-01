import os

base_path = r"C:\Users\davit\Downloads\dataset_for_arm_plates"
labels_path = os.path.join(base_path, "labels")

os.makedirs(labels_path, exist_ok=True)

extensions = (".jpg", ".jpeg", ".png")

for img_name in os.listdir(base_path):

    if not img_name.lower().endswith(extensions):
        continue

    # Plate text = filename without extension
    plate_text = os.path.splitext(img_name)[0]

    label_file = os.path.join(
        labels_path,
        plate_text + ".txt"
    )

    if os.path.exists(label_file):
        continue

    with open(label_file, "w", encoding="utf-8") as f:
        f.write(plate_text)

print("Labels created!")