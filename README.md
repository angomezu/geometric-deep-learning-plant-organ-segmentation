# 3D Plant Organ Segmentation via Geometric Deep Learning on Point Clouds

**Angel A. Barrera**  
Department of Mathematics & Statistics  
East Tennessee State University  

**Inhwan Jung**  
East Tennessee State University  

**Luke Hussung**  
East Tennessee State University  

In collaboration with **Oak Ridge National Laboratory (ORNL)**  
December 2025  

---

## Overview

This repository contains research code developed to study **semantic segmentation of 3D plant point clouds** using geometric deep learning techniques. The project focuses on separating biologically meaningful plant organs—**Stem, Leaf, and Support Stake**—from high-resolution LiDAR scans collected at the Advanced Plant Phenotyping Laboratory (APPL) at Oak Ridge National Laboratory.

The work was conducted as part of the graduate course **STAT 5920 – Internship Experience in Data Science II** at East Tennessee State University and represents an academic research collaboration rather than a deployed production system.

---

## Research Motivation

High-throughput plant phenotyping is critical for bioenergy and agricultural research, yet traditional 2D imaging approaches struggle to capture volumetric and structural traits of complex, woody plants. LiDAR-based 3D point clouds offer rich geometric information but introduce challenges related to:

- Geometric ambiguity between biological and abiotic structures  
- Severe class imbalance (e.g., stems vs. leaves)  
- Lack of spectral (RGB) information  
- Occlusion and sparse sampling at early growth stages  

This project investigates whether **explicit geometric feature engineering combined with Dynamic Edge Convolutional Neural Networks (DECNNs)** can effectively address these challenges.

---

## Scope and Limitations

**Important Notice**

- No data is included in this repository.  
- All raw and processed point cloud data, labels, and metadata provided by ORNL are excluded due to confidentiality and intellectual property restrictions.  
- The code is shared **for research and methodological demonstration purposes only**.

Any results shown here are conditional on the specific dataset, annotation protocol, and experimental setup described in the accompanying documentation.

---

## Methodology Summary

### Manual Annotation Protocol

A structured, rule-based manual annotation protocol was developed using **CloudCompare** to label individual point clouds into the following semantic classes:

- Stem  
- Leaf  
- Support Stake  
- Background  

The protocol emphasizes:

- Complete coverage (every point assigned a class)  
- Priority on accurate stem segmentation  
- Consistent naming conventions  
- Careful handling of organ overlap and occlusion  

A total of **30 point clouds** were fully annotated to support supervised learning.

---

### Feature Engineering

Raw XYZ coordinates alone were insufficient to distinguish flat leaves from cylindrical stems and stakes. To enrich the input representation, local geometric descriptors were computed using a k-nearest neighbor approach, including:

- Linearity  
- Planarity  
- Sphericity  
- Relative Height  

These features capture local shape characteristics critical for organ-level discrimination.

---

### Model Architecture

The primary model implemented is a **Dynamic Edge Convolutional Neural Network (DECNN)**, which:

- Constructs local neighborhood graphs dynamically at each layer  
- Learns edge features capturing local geometric relationships  
- Is well-suited for unstructured point cloud data  

To address severe class imbalance, a composite loss function combining **Weighted Cross-Entropy and Dice Loss** was used.

---

### Evaluation Strategy

Model performance was evaluated using:

- Intersection over Union (IoU)  
- Recall and Precision (per class)  
- Sample-averaged metrics  
- Bootstrapped confidence intervals  

Qualitative evaluation was performed via 3D visualizations of predicted segmentations to analyze structural reconstruction and failure modes.

---

## Repository Structure

```text
├── dataset.py        # Data loading, normalization, feature engineering
├── model.py          # Dynamic Edge CNN architecture definition
├── train.py          # Training loop and experiment configuration
├── evaluation.py     # Metric computation and bootstrapped evaluation
├── inference.py      # Inference and visualization pipeline
│
├── check_data.py     # Data integrity validation scripts
├── check_labels.py   # Label consistency checks
├── count_nans.py     # NaN and numerical stability audits
│
└── README.md

The codebase is intentionally modular, separating model logic, data handling, and execution scripts to promote reproducibility and safe experimentation.

```
---

## Results (Illustrative)

Under the described experimental setup, the model achieved:

- Strong segmentation performance on the dominant Leaf class  
- High recall for the critical Stem class despite severe class imbalance  
- Meaningful reconstruction of primary plant structure in qualitative analysis  

Observed limitations include stem–stake ambiguity and resolution-induced boundary artifacts, which are discussed in detail in the accompanying report.

---

## Future Research Directions

Potential extensions of this work include:

- Incorporation of RGB or multispectral data to reduce geometric ambiguity  
- Higher-resolution training enabled by improved hardware resources  
- Expanded datasets across growth stages and environmental conditions  
- Comparative studies with alternative point cloud architectures  

---

## Acknowledgments

This research used resources of the **Advanced Plant Phenotyping Laboratory** and the **Center for Bioenergy Innovation (CBI)**, which is a U.S. Department of Energy Bioenergy Research Center supported by the Office of Biological and Environmental Research in the DOE Office of Science. Oak Ridge National Laboratory is managed by **UT-Battelle, LLC** for the U.S. Department of Energy under Contract Number **DE-AC05-00OR22725**.

---

## Disclaimer

The views and conclusions contained in this repository are those of the authors and do not necessarily represent the views of Oak Ridge National Laboratory or the U.S. Department of Energy. The code is provided for academic and research purposes only.
