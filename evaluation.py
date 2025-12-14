import torch
import numpy as np
from torch_geometric.loader import DataLoader
from src.model import OakRidgeSegmenter
from src.dataset import OakRidgeDataset
from tqdm import tqdm
import os
import sys

# this script calculates evaluation metrics for the ORNL segmentation we trained

# configurations
# change this path as needed to point to your model and data
MODEL_PATH = "models/model_run4_nuclear(solid).pth"
DATA_ROOT = "data/val"
SPLIT = "val"
BATCH_SIZE = 1
NUM_CLASSES = 3
IN_CHANNELS = 7

# Class Map
CLASS_NAMES = {0: 'Stem', 1: 'Leaf', 2: 'Stake'}


def compute_confidence_interval(scores, num_bootstraps=1000, confidence_level=0.95):
    """
    Calculates the 95% Confidence Interval using Bootstrapping.
    scores: List or numpy array of IoU scores (one per sample).
    """
    scores = np.array(scores)
    if len(scores) < 2:
        return np.mean(scores), 0, 0

    bootstrapped_means = []
    rng = np.random.default_rng(seed=42)

    for _ in range(num_bootstraps):
        sample = rng.choice(scores, size=len(scores), replace=True)
        bootstrapped_means.append(np.mean(sample))

    lower_p = (1.0 - confidence_level) / 2.0 * 100
    upper_p = (1.0 + confidence_level) / 2.0 * 100
    lower_bound = np.percentile(bootstrapped_means, lower_p)
    upper_bound = np.percentile(bootstrapped_means, upper_p)

    return np.mean(scores), lower_bound, upper_bound


def calculate_metrics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating metrics on {device}")

    # 1. Load Data
    print(f"Loading {SPLIT} dataset from {DATA_ROOT}...")
    dataset = OakRidgeDataset(root=DATA_ROOT, split=SPLIT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    if len(dataset) == 0:
        print(f"Error: No data found in {DATA_ROOT}.")
        return

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    model = OakRidgeSegmenter(in_channels=IN_CHANNELS,
                              out_channels=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Global Confusion Matrix (for overall scores)
    global_cm = torch.zeros(NUM_CLASSES, NUM_CLASSES,
                            dtype=torch.long, device=device)
    # Per-Sample IoUs (for bootstrapping)
    sample_mean_ious = []

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)

            out = model(batch.x, batch.pos, batch.batch)
            preds = torch.argmax(out, dim=1)
            targets = batch.y

            # Vectorized CM for this batch/sample
            mask = (targets >= 0) & (targets < NUM_CLASSES)
            stacked = targets[mask] * NUM_CLASSES + preds[mask]
            bincount = torch.bincount(stacked, minlength=NUM_CLASSES**2)
            local_cm = bincount.reshape(NUM_CLASSES, NUM_CLASSES)

            # Updating the Global
            global_cm += local_cm

            # Calculate Local mIoU for this sample
            lcm = local_cm.cpu().numpy()
            sample_ious = []
            for i in range(NUM_CLASSES):
                tp = lcm[i, i]
                fp = np.sum(lcm[:, i]) - tp
                fn = np.sum(lcm[i, :]) - tp
                union = tp + fp + fn
                if union > 0:
                    sample_ious.append(tp / union)

            if sample_ious:
                sample_mean_ious.append(np.mean(sample_ious))

    # Compute & Print Metrics
    print("\n" + "="*85)
    print(f"{'Class':<12} | {'IoU (%)':<10} | {'Recall (%)':<12} | {'Precision (%)':<14} | {'Support'}")
    print("-" * 85)

    cm = global_cm.cpu().numpy()
    global_ious = []

    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        union = tp + fp + fn
        iou = tp / union if union > 0 else 0
        global_ious.append(iou)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        name = CLASS_NAMES.get(i, f"Class {i}")
        count = np.sum(cm[i, :])

        print(
            f"{name:<12} | {iou*100:<10.2f} | {recall*100:<12.2f} | {precision*100:<14.2f} | {count}")

    print("-" * 85)

    # Statistical Significance
    mean_iou = np.mean(global_ious)
    overall_acc = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0

    print(f"Global Mean IoU:   {mean_iou*100:.2f}%")
    print(f"Overall Accuracy:  {overall_acc*100:.2f}%")

    # Running the Bootstrap
    if len(sample_mean_ious) > 1:
        mean_boot, lower, upper = compute_confidence_interval(sample_mean_ious)
        print("Statistical Significance")
        print(f"Sample-Averaged mIoU: {mean_boot*100:.2f}%")
        print(f"Confidence Interval:  [{lower*100:.2f}% - {upper*100:.2f}%]")
    else:
        print("\nNot enough samples for bootstrap analysis.")

    print("="*85)


if __name__ == "__main__":
    calculate_metrics()
