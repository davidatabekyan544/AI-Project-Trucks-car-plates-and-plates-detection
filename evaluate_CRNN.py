import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score
from arm_plates_dataset_loader import PlateDataset, ALPHABET, collate_fn
from CRNN_model import CRNN
import numpy as np


def decode_text(indices, alphabet):
    """CTC Greedy Decoder: Removes blanks (0) and repeated characters."""
    res = ""
    for i, idx in enumerate(indices):
        if idx > 0:  # 0 is CTC Blank
            res += alphabet[idx - 1]
    return res


def calculate_cer(gt, pred):
    """Calculates the edit distance between ground truth and prediction."""
    n, m = len(gt), len(pred)
    if n == 0: return m
    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def evaluate_model(model_path, data_dir, label_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Evaluation on {device} ---")

    num_classes = len(ALPHABET) + 1
    model = CRNN(num_classes).to(device)

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = PlateDataset(data_dir, label_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    correct_plates = 0
    total_cer = 0
    y_true_all, y_pred_all = [], []

    print(f"Testing on {len(dataset)} images...\n")

    with torch.no_grad():
        for i, (imgs, labels, _) in enumerate(loader):
            imgs = imgs.to(device)
            logits = model(imgs)  # [W, B, C]

            # Greedy Decoding
            preds = torch.argmax(logits, 2)
            raw_indices = preds[:, 0].cpu().tolist()

            final_indices = []
            prev = -1
            for idx in raw_indices:
                if idx != 0 and idx != prev:
                    final_indices.append(idx)
                prev = idx

            gt_indices = labels.cpu().tolist()
            pred_str = decode_text(final_indices, ALPHABET)
            gt_str = decode_text(gt_indices, ALPHABET)

            if pred_str == gt_str:
                correct_plates += 1

            total_cer += calculate_cer(gt_str, pred_str)

            # Character-level metrics
            max_len = max(len(gt_indices), len(final_indices))
            y_true_all.extend(gt_indices + [0] * (max_len - len(gt_indices)))
            y_pred_all.extend(final_indices + [0] * (max_len - len(final_indices)))

            if i < 15:
                status = "✅" if pred_str == gt_str else "❌"
                print(f"Image {i + 1:02d} | GT: {gt_str:10} | Pred: {pred_str:10} | {status}")

    # Final Stats
    acc = (correct_plates / len(dataset)) * 100
    avg_cer = (total_cer / len(dataset)) * 100
    prec = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)

    print("\n" + "=" * 40)
    print(f"Plate-Level Accuracy: {acc:.2f}%")
    print(f"Avg Character Error:  {avg_cer:.2f}%")
    print(f"Character Precision:  {prec:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    # Point this to your original REAL test set to see true performance
    TEST_BASE = r"C:\Users\davit\Downloads\dataset_for_arm_plates\test"
    evaluate_model(
        "arm_plate_crnn.pth",
        os.path.join(TEST_BASE, "images"),
        os.path.join(TEST_BASE, "labels")
    )