# Plant Organ Segmentation from 3D LiDAR Point Clouds via Geometric Deep Learning

<div align="center">

### Authors

**[Angel A. Barrera-Gomez](https://www.linkedin.com/in/angomezu/)**,  
**[Inhwan Jung](https://www.linkedin.com/in/INHWAN_LINKEDIN/)**,  
**[Luke Hussung](https://www.linkedin.com/in/luke-hussung-4a2671252/)**  

Applied Data Science Program  
Department of Mathematics & Statistics  
East Tennessee State University  

**External Research Collaboration**  
Oak Ridge National Laboratory (ORNL)

</div>

<p align="center">
  <img src="assets/prediction.gif" alt="3D LiDAR scan of plant in controlled phenotyping environment" width="750"><br>
  <em>Figure 1:  Final model prediction after feature engineering and hyperparameter tuning, showing coherent reconstruction of plant organs 65.58% mIoU and 82% stem recall.</em>
</p>

---

## Table of Contents

- [Abstract](#abstract)
- [Contributions](#contributions)
- [Repository Purpose](#repository-purpose)
- [Tech Stack](#tech-stack)
- [Method Overview](#method-overview)
  - [Problem Setting](#problem-setting)
  - [Annotation Protocol](#annotation-protocol)
  - [Geometric Feature Engineering](#geometric-feature-engineering)
  - [Learning Architecture](#learning-architecture)
  - [Evaluation Protocol](#evaluation-protocol)
- [Results (Illustrative)](#results-illustrative)
- [Code Structure](#code-structure)
- [Reproducibility](#reproducibility)
- [Future Research Directions](#future-research-directions)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Disclaimer](#disclaimer)


---
## Abstract

Semantic segmentation of unstructured 3D point clouds remains a challenging problem, particularly in domains where appearance cues are unavailable. In plant phenotyping, LiDAR-based point clouds provide rich geometric information but suffer from class imbalance, occlusion, and geometric ambiguity between biological and abiotic structures.

This work investigates the use of **geometry-aware deep learning** for organ-level plant segmentation using **Dynamic Edge Convolutional Neural Networks (DECNNs)**. We propose a structured annotation protocol, geometric feature augmentation, and a loss formulation tailored to highly imbalanced plant data. The released code focuses on methodology and reproducibility and accompanies an ongoing research manuscript.

---

## Contributions

- Geometry-based semantic segmentation of plant organs from 3D LiDAR point clouds  
- A structured manual annotation protocol tailored to complex biological structures  
- Integration of local geometric descriptors within dynamic graph convolutional networks  
- Robust learning under severe class imbalance in organ-level segmentation tasks  
- A research-oriented, modular codebase designed to support reproducible experimentation

---

## Repository Purpose

This repository releases **research code** developed in collaboration with **Oak Ridge National Laboratory** for studying semantic segmentation of 3D plant point clouds.

**Important:**  
- No raw or processed data is included  
- No trained model checkpoints are provided  
- The repository focuses on **methodology, architecture, and evaluation**

---

## Tech Stack

### Programming Language
- Python

### Deep Learning & Graph Neural Networks
- PyTorch
- PyTorch Geometric (EdgeConv, radius-based dynamic graph construction)

### 3D Point Cloud Processing
- Open3D
- CloudCompare (manual annotation and ground truth generation)

### Feature Engineering & Geometry
- NumPy
- Eigenvalue-based geometric descriptors (surface normals, linearity, planarity, sphericity, relative height)

### Machine Learning & Post-processing
- scikit-learn (DBSCAN, PCA)

### Data Augmentation & Pipelines
- Torch Geometric Transforms (voxelization, rotation, scaling, jitter)

### Evaluation & Statistics
- IoU, Precision, and Recall metrics
- Bootstrap resampling (95% confidence intervals)

### Visualization & Media
- Open3D visualization

### Hardware Acceleration
- CUDA

---

## Method Overview

### Problem Setting

Given a 3D LiDAR point cloud acquired in a controlled phenotyping environment, the goal is to assign each point a semantic label corresponding to biologically meaningful plant structures:

- **Stem**
- **Leaf**
- **Support Stake**
- **Background**

The task is challenging due to:
- Severe class imbalance
- Structural similarity between stems and stakes
- Occlusion and sparse sampling
- Absence of RGB or spectral information

---

### Annotation Protocol

To construct reliable ground-truth labels for supervised learning, a structured, rule-based annotation protocol was developed using CloudCompare. The protocol was designed to ensure:

- Complete point-wise labeling of all visible structures  
- Priority on accurate stem delineation as the primary structural axis  
- Consistent handling of overlapping organs and geometric ambiguity  

A total of 30 fully annotated 3D LiDAR point clouds were generated and used for supervised training and evaluation.

---

### Geometric Feature Engineering

To overcome the limitations of raw XYZ coordinates, each point is augmented with local geometric descriptors computed within a fixed-radius neighborhood:

- Linearity  
- Planarity  
- Sphericity  
- Relative Height  

These features encode local shape properties critical for organ discrimination.

---

### Learning Architecture

The core model is a **Dynamic Edge Convolutional Neural Network (DECNN)** that:

- Dynamically constructs neighborhood graphs per layer
- Learns edge features capturing local geometry
- Operates directly on unstructured point clouds

To address extreme class imbalance, training uses a composite loss combining:

- Weighted Cross-Entropy  
- Dice Loss  

---

### Evaluation Protocol

Model performance is evaluated using:

- Intersection over Union (IoU)
- Precision and Recall (per class)
- Sample-averaged metrics
- Bootstrapped confidence intervals

Qualitative evaluation is performed via 3D visualization of predicted segmentations.

---

## Results (Illustrative)

Under the described experimental setup:

- Strong performance is observed on the dominant **Leaf** class
- High recall is achieved for the biologically critical **Stem** class
- Qualitative results show coherent reconstruction of plant structure

Limitations include stem–stake ambiguity and boundary artifacts due to resolution constraints.

---

## Code Structure

The repository is organized as a modular research pipeline:

```text
├── data/                # Directory structure only (no data included)
│   ├── train/
│   ├── val/
│   └── test/
│
├── src/
│   ├── dataset.py       # ETL and geometric feature computation
│   ├── model.py         # Dynamic Edge CNN architecture
│   └── inference.py     # Inference and post-processing
│
├── validations/         # Data integrity and sanity checks
│   ├── check_data.py
│   ├── check_labels.py
│   └── count_nans.py
│
├── train.py             # Training loop
├── evaluation.py        # Metric computation and bootstrapping
├── visualization.py    # 3D visualization utilities
└── README.md
```

All directories related to raw data, predictions, and model checkpoints are **intentionally excluded** from this repository to comply with data confidentiality and intellectual property constraints associated with Oak Ridge National Laboratory (ORNL).

---

### Reproducibility

Due to ORNL data restrictions, reproducing the full experimental results requires **authorized access** to the Advanced Plant Phenotyping Laboratory (APPL) dataset.

However, the codebase is written to be **dataset-agnostic** and can be readily adapted to alternative 3D point cloud datasets with compatible annotation formats and geometric structure.

---

### Future Research Directions

Potential extensions of this work include:

- Incorporation of RGB or multispectral information to reduce geometric ambiguity
- Evaluation of alternative point cloud architectures (e.g., PointNet++, KPConv, transformer-based models)
- Temporal modeling across plant growth stages
- Scalable experimentation via containerization and MLOps workflows (e.g., Docker, experiment tracking, automated evaluation pipelines)

---

### Citation

If you find this work useful in your research, please consider citing:

```bibtex
@unpublished{barrera2025plantseg,
  title  = {Plant Organ Segmentation from 3D LiDAR Point Clouds via Geometric Deep Learning},
  author = {Barrera-Gomez, Angel A. and Jung, Inhwan and Hussung, Luke},
  year   = {2025}
}
```

### Acknowledgments

This research used resources of the Advanced Plant Phenotyping Laboratory and the Center for Bioenergy Innovation (CBI), which is a U.S. Department of Energy Bioenergy Research Center supported by the Office of Biological and Environmental Research in the DOE Office of Science. Oak Ridge National Laboratory is managed by UT-Battelle, LLC for the U.S. Department of Energy under Contract Number DE-AC05-00OR22725.

We thank **Dr. John Lagergren**, **Dr. Larry M. York**, and **Anand Seethepalli** (Oak Ridge National Laboratory, Biosciences Division) for providing access to experimental data, domain expertise, and valuable feedback throughout the project.



---
### Disclaimer

The views and conclusions expressed in this repository are those of the authors and do not necessarily represent the views of Oak Ridge National Laboratory or the U.S. Department of Energy. The code is provided for **academic and research purposes only**.
