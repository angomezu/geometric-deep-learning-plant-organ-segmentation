## Reproducibility & Installation

### Continuous Integration and Code Quality

This repository enforces consistent coding standards and documentation to support long-term reproducibility and collaborative research.

All Python code is automatically checked using:

- **Ruff** for PEP8 and PEP257 compliance
- **Pre-commit hooks** to prevent non-compliant code from being committed locally
- **GitHub Actions CI** to validate code quality on every push and pull request

The CI pipeline runs the following checks:

```bash
ruff check .
ruff format --check .
```
Pull requests to the main branch are blocked unless all checks pass, ensuring that the repository remains clean, readable, and reproducible over time.

### Data Availability

Due to data access restrictions associated with Oak Ridge National Laboratory (ORNL), the original datasets used in this study are **not publicly available**. Full reproduction of the reported experimental results therefore requires **authorized access** to the Advanced Plant Phenotyping Laboratory (APPL) data.

That said, the codebase is **dataset-agnostic by design**. Any 3D LiDAR point cloud dataset can be used **provided that**:
- Point clouds are available in **XYZ format** (e.g., `.txt`, `.pcd`, `.ply`)
- Point-wise semantic labels are provided (or generated) following a compatible annotation scheme
- The data can be adapted to the expected input format used by the dataset loader

This enables reuse of the pipeline for **methodological experimentation**, architectural benchmarking, and extension to alternative 3D segmentation tasks.


### Installation

The main dependencies of the project are listed below.

**Core Requirements**
- Python ≥ 3.8
- CUDA ≥ 11.x (optional, but recommended for training)
- PyTorch + PyTorch Geometric
- Open3D


### Step 1: Clone the Repository and Create Environment

```bash
git clone https://github.com/angomezu/geometric-deep-learning-plant-organ-segmentation.git
cd geometric-deep-learning-plant-organ-segmentation

conda create -n plantseg python=3.9 pip
conda activate plantseg
```

### Step 2: Install PyTorch

Install PyTorch with CUDA support (adjust CUDA version if needed):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only usage:

```bash
pip install torch torchvision torchaudio
```

### Step 3: Install PyTorch Geometric

Install PyTorch Geometric and its dependencies:

```bash
pip install torch-geometric
```

If you encounter issues, refer to the official installation guide:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### Step 4: Install 3D Processing and ML Dependencies

```bash
pip install open3d numpy scikit-learn tqdm
```


## Notes on Usage

- Training scripts assume point-wise labeled data
- Data loaders and feature computation logic are implemented in src/dataset.py
- Visualization utilities require a functioning OpenGL context (for on-screen rendering)

### Users intending to apply the pipeline to new datasets may need to:

- Adapt the annotation format
- Update normalization statistics
- Adjust neighborhood radius and voxelization parameters

---

### Model checkpoints (.pth)

This project uses PyTorch checkpoint files (`.pth`) to store trained model weights.
Running `python train.py` will save a checkpoint to `models/` (see the filename in `train.py`).
Update `MODEL_PATH` (evaluation) and `CHECKPOINT` (visualization) to point to your `.pth`.

---

## Tech Stack

#### Programming Language
- Python

#### Deep Learning & Graph Neural Networks
- PyTorch
- PyTorch Geometric (EdgeConv, radius-based dynamic graph construction)

#### 3D Point Cloud Processing
- Open3D
- CloudCompare (manual annotation and ground truth generation)

#### Feature Engineering & Geometry
- NumPy
- Eigenvalue-based geometric descriptors (surface normals, linearity, planarity, sphericity, relative height)

#### Machine Learning & Post-processing
- scikit-learn (DBSCAN, PCA)

#### Data Augmentation & Pipelines
- Torch Geometric Transforms (voxelization, rotation, scaling, jitter)

#### Evaluation & Statistics
- IoU, Precision, and Recall metrics
- Bootstrap resampling (95% confidence intervals)

#### Visualization & Media
- Open3D visualization

#### Hardware Acceleration
- CUDA
