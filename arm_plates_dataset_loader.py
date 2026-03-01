import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Define alphabet based on Armenian plate characters
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_idx = {char: i + 1 for i, char in enumerate(ALPHABET)}

# Shared transformation for both Training and Testing
data_transform = transforms.Compose([
    transforms.Resize((64, 320)), # 320 width provides more 'slots' for letters
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Helps with lighting variety
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PlateDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform if transform else data_transform

        # Filter out broken or non-image files
        all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.img_names = []

        print(f"Verifying images in {img_dir}...")
        for f in all_files:
            img_path = os.path.join(img_dir, f)
            try:
                with Image.open(img_path) as img:
                    img.verify()
                self.img_names.append(f)
            except:
                print(f"Skipping corrupted image: {f}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        # Match image to .txt label
        label_file = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_file)

        with open(label_path, 'r', encoding='utf-8') as f:
            # Extracts the plate string (e.g., "02DA012") from the file
            label_str = f.read().strip().split()[-1]

        encoded_label = [char_to_idx[c] for c in label_str if c in char_to_idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.LongTensor(encoded_label), torch.LongTensor([len(encoded_label)])


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    return torch.stack(images), torch.cat(labels), torch.cat(lengths)