import os
import shutil


def merge_datasets(real_path, syn_path, output_path):
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)

    def copy_files(src_dir, prefix):
        img_src = os.path.join(src_dir, "images")
        lbl_src = os.path.join(src_dir, "labels")

        for file in os.listdir(img_src):
            # Rename with prefix to avoid filename conflicts
            new_name = prefix + "_" + file
            shutil.copy(os.path.join(img_src, file), os.path.join(output_path, "images", new_name))

            # Copy corresponding label
            label_file = file.replace(".png", ".txt").replace(".jpg", ".txt")
            if os.path.exists(os.path.join(lbl_src, label_file)):
                shutil.copy(os.path.join(lbl_src, label_file),
                            os.path.join(output_path, "labels", prefix + "_" + label_file))

    print("Merging Real Data...")
    copy_files(real_path, "real")
    print("Merging Synthetic Data...")
    copy_files(syn_path, "syn")
    print(f"✅ Done! Total images in {output_path}: {len(os.listdir(os.path.join(output_path, 'images')))}")


# Run it
REAL = r"C:\Users\davit\Downloads\dataset_for_arm_plates\train"
SYN = r"C:\Users\davit\Downloads\synthetic_plates"
OUT = r"C:\Users\davit\Downloads\combined_dataset"
merge_datasets(REAL, SYN, OUT)