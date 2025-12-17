"""Train the point-cloud segmentation model with weighted CE + Dice loss."""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.dataset import OakRidgeDataset
from src.model import OakRidgeSegmenter

# This adds the repository root to PYTHONPATH so `src/` imports work when running this.
# Script directly (e.g., `python train.py`) without installing the package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# This script trains the ORNL segmentation model using a combination of
# Cross-Entropy loss and Dice loss.
# Configuration
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 100
# Stem, Leaf, Stake (No Background since this class affects performance)
NUM_CLASSES = 3
IN_CHANNELS = 7


# Dice Loss Implementation
class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation."""

    def __init__(self, smooth=1e-6):
        """Initialize Dice loss.

        Args:
            smooth: Small constant to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """Compute Dice loss from logits and integer class targets."""
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()

        # Calculate intersection and union
        intersection = torch.sum(probs * targets_one_hot, dim=0)
        cardinality = torch.sum(probs + targets_one_hot, dim=0)

        # Dice score per class: 2*TP / (Pred + Truth)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Loss is 1 - Dice Score (Average over classes)
        return 1.0 - torch.mean(dice_score)


def train():
    """Run the training loop and save the best checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = OakRidgeDataset(root="data/train", split="train")
    val_dataset = OakRidgeDataset(root="data/val", split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = OakRidgeSegmenter(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Cross Entropy Weights (Balanced for 3 classes)
    # This is because Stem(0) and Stake(2) are rare compared to Leaf(1)
    ce_weights = torch.tensor([5.0, 1.0, 5.0]).to(device)
    criterion_ce = nn.CrossEntropyLoss(weight=ce_weights)

    # Dice Loss
    criterion_dice = DiceLoss()
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_accum = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.pos, batch.batch)

            # Combined Loss
            loss_ce = criterion_ce(out, batch.y)
            loss_dice = criterion_dice(out, batch.y)
            loss = loss_ce + loss_dice

            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.pos, batch.batch)

                l_ce = criterion_ce(out, batch.y)
                l_dice = criterion_dice(out, batch.y)
                val_loss += (l_ce + l_dice).item()

        avg_loss = val_loss / len(val_loader)
        train_avg = train_loss_accum / len(train_loader)
        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_avg:.4f} | "
            f"Val Loss = {avg_loss:.4f}"
        )

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            # we can change this name as needed
            torch.save(model.state_dict(), "models/model_run4_nuclear.pth")
            print(f"  Saved Best Model (Loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()
