"""Neural network model definition for point-cloud semantic segmentation."""

import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import radius_graph


class OakRidgeSegmenter(nn.Module):
    """Dynamic graph CNN (EdgeConv) segmenter for plant organ labels."""

    def __init__(self, in_channels=7, out_channels=4, r=0.05, k=20):
        """Initialize the segmenter model.

        Args:
            in_channels: Number of input features per point.
            out_channels: Number of output classes.
            r: Radius used for radius-graph neighborhood construction.
            k: Max neighbors per node within the radius.
        """
        super().__init__()
        self.r = r
        self.k = k

        # Layer 1: Input (in_channels) -> 64
        # EdgeConv concatenates x_i and (x_j - x_i), so MLP sees 2 * in_channels.

        self.mlp1 = nn.Sequential(
            nn.Linear(2 * in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv1 = EdgeConv(self.mlp1, aggr="max")

        # Layer 2: 64 -> 128
        self.mlp2 = nn.Sequential(
            nn.Linear(2 * 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv2 = EdgeConv(self.mlp2, aggr="max")

        # Layer 3: 128 -> 256
        self.mlp3 = nn.Sequential(
            nn.Linear(2 * 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.conv3 = EdgeConv(self.mlp3, aggr="max")
        """
        This is the decoder/segmentation head
        Here, we concatenate skip connections (64 + 128 + 256) -> 448
        This allows the model to use fine details (Layer 1) and global 
        context (Layer 3) simultaneously.
        """
        self.head = nn.Sequential(
            nn.Linear(448, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # Regularization to prevent overfitting on small data
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_channels),
        )

    def forward(self, x, pos, batch):
        """Build a radius graph and run EdgeConv layers.

        Uses radius-based neighborhood construction to avoid connecting distant points.
        """
        edge_index = radius_graph(
            pos, r=self.r, batch=batch, max_num_neighbors=self.k, loop=False
        )

        # Graph Convolutions
        # We pass the dynamic edge_index to EdgeConv

        # Conv 1
        x1 = self.conv1(x, edge_index)

        # Conv 2
        x2 = self.conv2(x1, edge_index)

        # Conv 3
        x3 = self.conv3(x2, edge_index)

        # Feature Concatenation
        # Combine local (x1) and global/complex (x3) features
        out = torch.cat([x1, x2, x3], dim=1)

        # Classification
        return self.head(out)
