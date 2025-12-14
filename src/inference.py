import os
import sys
import torch
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from model import OakRidgeSegmenter

# Configuration
IN_CHANNELS = 7
NUM_CLASSES = 3
STAKE_CLASS_IDX = 2

# Refinement Parameters
DBSCAN_EPS = 15.0
DBSCAN_MIN_SAMPLES = 10
CYLINDER_RADIUS = 7.0
MAX_PCA_ITERATIONS = 20
CONVERGENCE_THRESH = 0.01


def compute_features(points):
    """
    Replicates the feature engineering from dataset.py.
    Returns: Tensor [N, 7] (Nx, Ny, Nz, Lin, Plan, Sph, Height)
    """
    num_points = points.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Normals (Radius 0.1, MaxNN 30)
    # In here, we normalize for feature calc to match training distribution
    XYZ_MEAN = np.array([6.408, 12.371, 210.991], dtype=np.float32)
    XYZ_STD = np.array([122.843, 115.391, 260.731], dtype=np.float32)

    points_norm = (points - XYZ_MEAN) / XYZ_STD

    pcd_norm = o3d.geometry.PointCloud()
    pcd_norm.points = o3d.utility.Vector3dVector(points_norm)

    # Now, we use the params from dataset.py (radius=0.1)
    pcd_norm.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_norm.orient_normals_consistent_tangent_plane(k=20)
    normals = np.asarray(pcd_norm.normals, dtype=np.float32)

    pcd_norm.estimate_covariances(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    covariances = np.asarray(pcd_norm.covariances)
    eigenvalues = np.linalg.eigvalsh(covariances)[:, ::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-12)

    l1, l2, l3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]

    linearity = np.clip((l1 - l2) / l1, 0, 1)
    planarity = np.clip((l2 - l3) / l1, 0, 1)
    sphericity = np.clip(l3 / l1, 0, 1)

    z_norm = points_norm[:, 2]
    relative_height = (z_norm - z_norm.min()) / \
        (z_norm.max() - z_norm.min() + 1e-6)

    features = np.column_stack((
        normals,
        linearity,
        planarity,
        sphericity,
        relative_height
    )).astype(np.float32)

    return torch.from_numpy(features), torch.from_numpy(points_norm)


def fit_line_pca(points):
    if points.shape[0] < 5:
        return None, None
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    direction = evecs[:, np.argmax(evals)]
    return centroid, direction


def distance_to_line(points, line_point, line_dir):
    vecs = points - line_point
    proj = np.dot(vecs, line_dir)
    closest = line_point + np.outer(proj, line_dir)
    dists = np.linalg.norm(points - closest, axis=1)
    return dists, proj


def refine_stake_predictions(points, preds):
    stake_mask = (preds == STAKE_CLASS_IDX)
    stake_points = points[stake_mask]

    if len(stake_points) < DBSCAN_MIN_SAMPLES:
        return stake_mask

    clustering = DBSCAN(
        eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(stake_points)
    labels = clustering.labels_

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    if not unique_labels:
        return stake_mask

    largest_cluster_lab = max(unique_labels, key=lambda x: np.sum(labels == x))
    cluster_mask = (labels == largest_cluster_lab)

    active_points = stake_points[cluster_mask]
    centroid, direction = fit_line_pca(active_points)

    if centroid is None:
        return stake_mask

    final_mask = np.zeros(len(points), dtype=bool)
    dists, _ = distance_to_line(points, centroid, direction)
    current_cylinder_mask = (dists < CYLINDER_RADIUS)

    for i in range(MAX_PCA_ITERATIONS):
        cyl_points = points[current_cylinder_mask]
        new_cent, new_dir = fit_line_pca(cyl_points)

        if new_cent is None:
            break

        dot_prod = np.abs(np.dot(direction, new_dir))
        angle = np.degrees(np.arccos(np.clip(dot_prod, 0, 1)))

        centroid, direction = new_cent, new_dir

        dists, projs = distance_to_line(points, centroid, direction)
        orig_stake_projs = projs[stake_mask]
        min_proj, max_proj = np.min(orig_stake_projs), np.max(orig_stake_projs)

        current_cylinder_mask = (dists < CYLINDER_RADIUS) & (
            projs > min_proj) & (projs < max_proj)

        if angle < CONVERGENCE_THRESH:
            break

    return current_cylinder_mask


def predict(model_path, input_file, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- INFERENCE on {input_file} using {device} ---")

    if input_file.endswith('.txt'):
        data_arr = np.loadtxt(input_file)
        points = data_arr[:, 0:3].astype(np.float32)
    elif input_file.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(input_file)
        points = np.asarray(pcd.points, dtype=np.float32)
    else:
        print("Unsupported file format.")
        return

   # Computing features
    features, points_norm_tensor = compute_features(points)

    batch = torch.zeros(points.shape[0], dtype=torch.long)

   # Loading the Model
    model = OakRidgeSegmenter(in_channels=IN_CHANNELS,
                              out_channels=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prediction
    with torch.no_grad():
        x = features.to(device)
        pos = points_norm_tensor.to(device)
        b = batch.to(device)

        logits = model(x, pos, b)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Refining Stake (Class 2)
    refined_stake_mask = refine_stake_predictions(points, preds)

    final_preds = preds.copy()
    final_preds[refined_stake_mask] = STAKE_CLASS_IDX

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_file).split('.')[0]
    out_path = os.path.join(output_dir, f"{base_name}_prediction.pcd")

    colors = np.zeros_like(points)
    colors[final_preds == 0] = [0, 1, 0]  # Stem
    colors[final_preds == 1] = [0, 0, 1]  # Leaf
    colors[final_preds == 2] = [1, 0, 0]  # Stake

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(points)
    out_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(out_path, out_pcd)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    MODEL_PATH = "models/model_run4_nuclear.pth"
    # Update this when neeeded to test other files you have.
    INPUT_FILE = "data/test/99920240603160423_labels.txt"
    OUTPUT_DIR = "predictions"

    if os.path.exists(INPUT_FILE) and os.path.exists(MODEL_PATH):
        predict(MODEL_PATH, INPUT_FILE, OUTPUT_DIR)
    else:
        print("Please set INPUT_FILE and MODEL_PATH in the script.")
