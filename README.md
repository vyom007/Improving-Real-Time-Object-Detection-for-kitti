
# Improving YOLOv5 for the KITTI Dataset with Ghost Convolutions and Coordinate Attention

**Authors:** [Vyom Sagar](https://vyomsagar.vercel.app), [Shikhar Kapoor](https://www.linkedin.com/in/shikhar-kapoor-thankyou/)  
**Affiliation:** University of Alabama at Birmingham  
**Emails:** vsagar@uab.edu, skapoor2@uab.edu  
**Mentor:** [Dr. Qing Tian](https://qtianreal.github.io/)

## Overview

This project investigates **architectural improvements to YOLOv5s** for real-time object detection on the **KITTI Object Detection dataset**.  
The goal is to improve **computational efficiency** while maintaining **competitive detection accuracy**, a key requirement for **autonomous driving and embedded deployment**.

We evaluate the impact of:
- **Ghost Convolutions** (lightweight feature generation)
- **Coordinate Attention (CA)** (spatially-aware attention)
- A **Hybrid Ghost + CA model**

Four model variants are compared:
1. **Baseline YOLOv5s**
2. **Ghost-only YOLOv5s**
3. **Coordinate Attention–only YOLOv5s**
4. **Hybrid Ghost + CA YOLOv5s**

This repository contains **training scripts, dataset preparation utilities, inference code, benchmarking tools, and architecture visualizations** used in the experiments reported in our project report :
---

## Motivation

YOLO-based detectors provide an excellent balance between speed and accuracy, but **standard convolution-heavy backbones** can be inefficient for deployment on:
- embedded automotive hardware  
- mobile and edge devices  
- real-time robotics systems  

This project explores whether **lightweight convolution modules** and **attention mechanisms** can reduce **parameters and FLOPs** while preserving detection performance on challenging real-world driving scenes.

---

## Model Variants

### 1. [Baseline YOLOv5s](https://vyom007.github.io/Improving-Real-Time-Object-Detection-for-kitti/yolov5_arch.html)
- CSPDarknet backbone  
- PANet neck (P3, P4, P5)  
- SPPF module  
- ~7.03M parameters, ~15.8 GFLOPs (640×640)

### 2. [Ghost-only YOLOv5s](https://vyom007.github.io/Improving-Real-Time-Object-Detection-for-kitti/yolov5s_kitti_ghost_only.html)
- Replaces selected standard convolutions with **Ghost Modules**
- Reduces redundant feature generation
- Strong accuracy–efficiency trade-off

### 3. [Coordinate Attention (CA) YOLOv5s](https://vyom007.github.io/Improving-Real-Time-Object-Detection-for-kitti/yolov5s_kitti_ca_only.html)
- Injects **Coordinate Attention blocks** into the backbone
- Improves spatial localization
- Lower pretrained-weight compatibility

### 4. [Hybrid Ghost + CA YOLOv5s](https://vyom007.github.io/Improving-Real-Time-Object-Detection-for-kitti/yolov5s_kitti_ghost_ca_.html)
- Combines lightweight convolutions with spatial attention
- Most parameter-efficient
- Lowest transfer-learning compatibility

Architecture visualizations for all variants are included as interactive HTML files.

---

## [Dataset](https://www.cvlibs.net/datasets/kitti/)

### KITTI Object Detection Dataset
- **7,481 annotated images**
- **8 classes:** Car, Van, Truck, Pedestrian, Cyclist, Tram, Misc, Person Sitting
- Real-world urban driving scenes
- Significant occlusion, truncation, and scale variation

### Preprocessing
- Images resized to **640 × 640**
- Standard YOLO augmentations:
  - Mosaic
  - Scaling
  - Flipping
  - Color jitter
- Annotations converted from KITTI format to YOLO `.txt` format

### Splits
- ~5,000 training images  
- ~1,500 validation images  

---

## Training Setup

- **Epochs:** 100  
- **Optimizer:** SGD with warm restarts  
- **Momentum:** 0.937  
- **Weight decay:** 5e-4  
- **Initial LR:** 0.01 (cosine schedule)  
- **Activation:** SiLU  
- **Initialization:** COCO-pretrained weights for all compatible layers  

New layers introduced by Ghost Modules or Coordinate Attention are **randomly initialized**, which directly impacts convergence behavior and final accuracy 

---

## Quantitative Results

**All models trained at 640×640 resolution under identical settings**

| Model Variant | Params (M) | GFLOPs | Precision | Recall | mAP@0.5 |
|--------------|-----------:|-------:|----------:|-------:|--------:|
| YOLOv5s (Baseline) | 7.03 | 15.8 | 0.915 | 0.863 | **0.910** |
| Ghost-only | 6.07 | 13.9 | 0.900 | 0.850 | 0.908 |
| CA-only | 7.35 | 17.1 | 0.913 | 0.856 | 0.908 |
| Ghost + CA | 6.39 | 15.2 | 0.906 | 0.823 | 0.892 |

### Key Observations
- **Ghost-only** reduces parameters by ~14% and GFLOPs by ~12% with almost no accuracy loss
- **Coordinate Attention** improves localization but increases computational cost
- **Hybrid Ghost + CA** achieves the highest efficiency but suffers reduced recall and mAP due to lower pretrained-weight reuse

---

## Transfer Learning Compatibility

| Model | Loaded / Total Layers | Compatibility | mAP@0.5 |
|-----|----------------------:|--------------:|--------:|
| Baseline | 342 / 349 | 98% | 0.910 |
| Ghost-only | 294 / 397 | 74% | 0.908 |
| CA-only | 246 / 458 | 54% | 0.908 |
| Ghost + CA | 198 / 506 | 39% | 0.892 |

Lower compatibility directly correlates with **slower convergence and reduced final accuracy** 

---

## Repository Structure

```text
.
├── yolov5/                         # YOLOv5 codebase
├── runs/train/                     # Training logs and checkpoints
├── kitti.yaml                      # Dataset configuration
├── prepare_kitti.py                # KITTI → YOLO conversion
├── train_kitti_full.py             # Training pipeline
├── infer.py                        # Inference script
├── benchmark_code.py               # FPS / latency benchmarking
├── test_train.py                   # Sanity checks
├── yolov5_arch.html                # Architecture visualization
├── yolov5s_kitti_ghost_only.html
├── yolov5s_kitti_ca_only.html
├── yolov5s_kitti_ghost_ca_.html
└── yolov5s_coco_pretrained_weights.pt
````

---

## Running the Code

### Dataset Preparation

```bash
python prepare_kitti.py
```

### Training

```bash
python train_kitti_full.py
```

### Inference

```bash
python infer.py
```

### Benchmarking

```bash
python benchmark_code.py
```

---

## Key Insights

* **Ghost Modules** provide the best accuracy–efficiency balance
* **Coordinate Attention** improves spatial reasoning but disrupts transfer learning
* **Hybrid models** require better initialization or training strategies to fully realize their potential

---

## Future Work

* Knowledge distillation to recover accuracy in CA-based models
* Evaluation on embedded devices (Jetson Nano / Xavier)
* Exploring alternative attention mechanisms (CBAM, ECA, SimAM)
* Deeper integration of CA into PANet or SPPF modules

---

## Real-World Impact

This work demonstrates that **carefully designed lightweighting strategies** can enable **real-time perception** for autonomous vehicles and embedded systems while maintaining strong detection accuracy .

---

