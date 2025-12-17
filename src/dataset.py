"""Dataset utilities and geometric feature computation for point-cloud segmentation."""

import os

import numpy as np
import open3d as o3d
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class OakRidgeDataset(Dataset):
    """PyTorch Geometric dataset for labeled plant LiDAR point clouds."""

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """Initialize the dataset and configure transforms for the given split."""
        self.split = split
        voxel_size = 0.02

        if transform is None and split == "train":
            self.transform = T.Compose(
                [
                    T.GridSampling(size=voxel_size),
                    T.RandomRotate(degrees=360, axis=2),
                    T.RandomScale((0.85, 1.15)),
                    T.RandomFlip(axis=0),
                    T.RandomFlip(axis=1),
                    T.RandomJitter(0.01),
                ]
            )
        elif transform is None:
            self.transform = T.GridSampling(size=voxel_size)
        else:
            self.transform = transform

        super().__init__(root, self.transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """List raw filenames expected under the dataset raw directory."""
        raw_dir = os.path.join(self.root, "raw")
        if not os.path.exists(raw_dir):
            return []
        return sorted([f for f in os.listdir(raw_dir) if f.endswith("_labels.txt")])

    @property
    def processed_file_names(self):
        """List processed filenames saved under the processed directory."""
        return [f"data_{self.split}_{i}.pt" for i in range(len(self.raw_file_names))]

    def len(self):
        """Return the number of processed samples."""
        return len(self.processed_file_names)

    def get(self, idx):
        """Load and return one processed sample by index."""
        path = os.path.join(self.processed_dir, f"data_{self.split}_{idx}.pt")
        return torch.load(path)

    def process(self):
        """Process raw point clouds into PyTorch Geometric Data objects."""
        print(f"Processing {self.split} data (Removing Background)...")
        idx = 0

        # Normalization Stats (Global)
        XYZ_MEAN = np.array([6.408, 12.371, 210.991], dtype=np.float32)
        XYZ_STD = np.array([122.843, 115.391, 260.731], dtype=np.float32)

        for filename in tqdm(self.raw_file_names):
            try:
                data_arr = np.loadtxt(os.path.join(self.root, "raw", filename))
            except (OSError, ValueError):
                continue

            points = data_arr[:, 0:3].astype(np.float32)
            labels = data_arr[:, -1].astype(np.int64)

            # Filtering the background (Class 3)
            mask = labels != 3
            points = points[mask]
            labels = labels[mask]

            # If the file was only background, skip it
            if len(points) < 10:
                print(f"Skipping {filename} (empty after background removal)")
                continue

            # Normalizing Coordinates
            points_norm = (points - XYZ_MEAN) / XYZ_STD

            # Feature Engineering part
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_norm)

            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=20)
            normals = np.asarray(pcd.normals, dtype=np.float32)

            pcd.estimate_covariances(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
            )
            evals = np.linalg.eigvalsh(np.asarray(pcd.covariances))[:, ::-1]
            evals = np.maximum(evals, 1e-12)
            l1, l2, l3 = evals[:, 0], evals[:, 1], evals[:, 2]

            linearity = np.clip((l1 - l2) / l1, 0, 1)
            planarity = np.clip((l2 - l3) / l1, 0, 1)
            sphericity = np.clip(l3 / l1, 0, 1)

            z_norm = points_norm[:, 2]
            rel_height = (z_norm - z_norm.min()) / (z_norm.max() - z_norm.min() + 1e-6)

            # Stack 7 features
            feats = np.column_stack(
                (normals, linearity, planarity, sphericity, rel_height)
            ).astype(np.float32)

            data = Data(
                pos=torch.from_numpy(points_norm),
                x=torch.from_numpy(feats),
                y=torch.from_numpy(labels).long(),
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(
                data, os.path.join(self.processed_dir, f"data_{self.split}_{idx}.pt")
            )
            idx += 1
        print(f"Saved {idx} files.")
