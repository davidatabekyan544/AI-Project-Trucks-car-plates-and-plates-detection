import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from arm_plates_dataset_loader import PlateDataset, ALPHABET, collate_fn
from CRNN_model import CRNN
import os


def train():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Hyperparameters
    base_path = r"C:\Users\davit\Downloads\combined_dataset"
    batch_size = 8  # Small batch for RTX 3050 stability
    lr = 0.00003  # Very low learning rate to prevent NaN
    epochs = 150
    num_classes = len(ALPHABET) + 1  # +1 for CTC Blank

    # 3. Data Loaders
    train_dataset = PlateDataset(os.path.join(base_path, "images"),
                                 os.path.join(base_path, "labels"))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 4. Model, Loss, Optimizer
    model = CRNN(num_classes).to(device)

    # zero_infinity=True is the secret to stopping NaN
    # zero_infinity is crucial for CTC Loss stability
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for imgs, labels, label_lengths in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)  # [W, B, C]

            # T = Width, N = Batch
            T, N, C = logits.size()
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)

            loss = criterion(logits.log_softmax(2), labels, input_lengths, label_lengths)

            if torch.isnan(loss):
                print("Skipping NaN loss batch...")
                continue

            loss.backward()

            # GRADIENT CLIPPING: Prevents the "Monster" jumps in math
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), "arm_plate_crnn.pth")

    print("Training Complete!")


if __name__ == "__main__":
    train()