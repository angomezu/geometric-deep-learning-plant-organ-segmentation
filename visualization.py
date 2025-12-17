"""Visualize point-cloud segmentation predictions."""

import numpy as np
import open3d as o3d
import torch

from src.dataset import OakRidgeDataset
from src.model import OakRidgeSegmenter

# CONFIGURATION
NUM_CLASSES = 4
# Change this path as needed
CHECKPOINT = r"C:\Oak Ridge\Model\models\model_run4_nuclear.pth"


def visualize_prediction():
    """Load a checkpoint and visualize predicted labels as colored points."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {CHECKPOINT}...")

    # First load Model (previous checkpoint)
    model = OakRidgeSegmenter(in_channels=4, out_channels=NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=device, weights_only=True)
    )
    model.eval()

    # Loading the Validation Data
    # We want to see how it performs on data it hasn't trained on heavily
    dataset = OakRidgeDataset(root="data", split="val")
    print(f"Found {len(dataset)} validation samples.")

    # Pick the first one (or change index to see others)
    data = dataset[0].to(device)

    # Running inference
    with torch.no_grad():
        # Pass x, pos, and batch (dummy batch of zeros since it's 1 plant)
        batch_vec = torch.zeros(data.pos.shape[0], dtype=torch.long).to(device)
        logits = model(data.x, data.pos, batch_vec)
        preds = logits.argmax(dim=1).cpu().numpy()

    # 4. Prepare Visualization
    points = data.pos.cpu().numpy()
    labels = data.y.cpu().numpy()

    # Define colors: 0=Stem, 1=Leaf, 2=Stake, 3=Background.
    # Update this mapping if your class indices differ.
    color_map = np.array(
        [
            [0.6, 0.6, 0.6],  # 0: Stem (Grey)
            [0.6, 0.4, 0.2],  # 1: Leaf (Brown)
            [0.0, 1.0, 0.0],  # 2: Stake (Green)
            [1.0, 0.0, 0.0],  # 3: Background (Red)
        ]
    )

    # Left: Ground Truth (What it actually is)
    pcd_true = o3d.geometry.PointCloud()
    pcd_true.points = o3d.utility.Vector3dVector(points)
    pcd_true.colors = o3d.utility.Vector3dVector(color_map[labels])
    # Shift it left
    pcd_true.translate([-200, 0, 0])

    # Right: Prediction (What the model thinks)
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points)
    pcd_pred.colors = o3d.utility.Vector3dVector(color_map[preds])
    # Shift it right
    pcd_pred.translate([200, 0, 0])

    print("LEFT:  Ground Truth (Reality)")
    print("RIGHT: Prediction (Model)")

    o3d.visualization.draw_geometries([pcd_true, pcd_pred])


if __name__ == "__main__":
    visualize_prediction()
