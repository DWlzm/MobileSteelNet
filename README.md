# MobileSteelNet: Steel Surface Defect Classification

MobileSteelNet is a lightweight deep learning network for steel surface defect classification.
This repository is the official code implementation of the paper and contains the **MobileSteelNet model only**.

---

## Table of Contents

- [Citation](#citation)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Network Architecture](#network-architecture)
- [The Two Key Innovations](#the-two-key-innovations)
- [Tensor Tests](#tensor-tests)
- [Training Outputs](#training-outputs)
- [Notes](#notes)

---

## Citation

```bibtex
@Article{s26031022,
AUTHOR = {Zou, Xiang and Liu, Zhongming and Xu, Chengjun and Zhang, Jiawei and Li, Zhaoyu},
TITLE = {MobileSteelNet: A Lightweight Steel Surface Defect Classification Network with Cross-Interactive Efficient Multi-Scale Attention},
JOURNAL = {Sensors},
VOLUME = {26},
YEAR = {2026},
NUMBER = {3},
ARTICLE-NUMBER = {1022},
URL = {https://www.mdpi.com/1424-8220/26/3/1022},
ISSN = {1424-8220},
ABSTRACT = {Steel surface defect classification is critical for industrial quality control, yet existing methods struggle to balance accuracy and efficiency for real-time deployment in vision-based sensor systems. This paper presents MobileSteelNet, a lightweight deep learning framework that introduces two novel modules: multi-scale feature fusion (MSFF), for integrating multi-stage features; and Cross-Interactive Efficient Multi-Scale Attention (CIEMA), which unifies inter-channel interaction, parallel multi-scale spatial extraction, and grouped efficient computation. Experiments on the NEU-DET dataset demonstrate that MobileSteelNet achieves 91.36% average accuracy, surpassing ResNet-50 (88.01%) and lightweight networks, including MobileNetV2 (86.08%). Notably, it achieves 93.70% accuracy on Scratch-type defects, representing an 82.12 percentage point improvement over baseline MobileNetV1. With a model size of only 8.2 MB, MobileSteelNet maintains superior performance while meeting lightweight deployment requirements, making it suitable for edge deployment in vision sensor systems for steel manufacturing.},
DOI = {10.3390/s26031022}
}
```

> If you wish to use this code, please send an email to **zhongmingliu2004@qq.com** to obtain the paper citation.

---

## Environment Setup

### 1. Requirements

| Item | Requirement |
| --- | --- |
| Python | >= 3.8 (3.8 – 3.12 recommended) |
| PyTorch | >= 1.9.0 |
| OS | Windows / Linux |
| Hardware | CUDA-enabled GPU (recommended); CPU also works |

### 2. Create a Virtual Environment (Recommended)

```bash
# conda
conda create -n mobilesteelnet python=3.10 -y
conda activate mobilesteelnet

# or venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
cd Code
pip install -r requirements.txt
```

The contents of `Code/requirements.txt`:

```
torch>=1.9.0
torchvision>=0.10.0
thop>=0.1.1
pandas>=1.3.0
numpy>=1.21.0
```

For GPU acceleration, install PyTorch following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/) according to your CUDA version, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> Note: `scikit-learn` is used for data splitting and evaluation metrics, and `matplotlib` is used for plotting. If either is missing at runtime, install them as well:
> ```bash
> pip install scikit-learn matplotlib
> ```

---

## Project Structure

```
MobileSteelNet-main/
├── README.md                        # Project documentation (this file)
├── Data/                            # Dataset root directory
│   └── NEU-DET.zip                  # Original archive (auto-extraction supported)
└── Code/
    ├── models/                      # Model definitions
    │   ├── __init__.py              # Exports MobileSteelNet only
    │   └── MobileSteelNet.py        # Full MobileSteelNet implementation (MSFF, CIEMA, tensor tests)
    ├── dataset/                     # Dataset handling
    │   ├── __init__.py
    │   ├── steel_dataset.py         # Dataset class and DataLoader construction
    │   └── transforms.py            # Data transforms and augmentation
    ├── utils/                       # Utilities
    │   ├── __init__.py
    │   ├── metrics.py               # Evaluation metrics (confusion matrix, precision, recall, F1)
    │   └── visualization.py         # Training curves, confusion matrix, and other plots
    ├── train.py                     # Training script (MobileSteelNet only)
    ├── requirements.txt             # Dependencies
    └── README.md                    # Sub-directory documentation
```

> The `models/` directory contains **only** `MobileSteelNet.py`. All core modules (depthwise separable convolution, MSFF, CIEMA) are self-contained in that single file and do not depend on any other model files.

---

## Dataset Preparation

### NEU-DET Directory Layout

Simply place `NEU-DET.zip` into the `Data/` directory and it will be extracted automatically; alternatively, extract it manually as follows:

```
Data/
├── NEU-DET.zip                # Original archive (optional, auto-extraction supported)
└── NEU-DET/                   # Extracted NEU-DET dataset directory
    ├── crazing/               # Crazing defect images
    ├── inclusion/             # Inclusion defect images
    ├── patches/               # Patches defect images
    ├── pitted/                # Pitted_surface defect images
    ├── rolled-in/             # Rolled_in_scale defect images
    └── scratches/             # Scratches defect images
```

There are 6 defect classes with 300 images each.

### Class Mapping

| Index | Class Name | Folder |
| --- | --- | --- |
| 0 | Crazing | `crazing` |
| 1 | Inclusion | `inclusion` |
| 2 | Patches | `patches` |
| 3 | Pitted_surface | `pitted` |
| 4 | Rolled_in_scale | `rolled-in` |
| 5 | Scratches | `scratches` |

> The code also supports the `FSSD-12` dataset (12 classes). Just change `dataset_type` in `train.py` to `'FSSD-12'`.

---

## Training

### 1. Set the Data Path

Open `Code/train.py` and change `data_dir` in `main()` to the **absolute path** of your local `Data/` directory:

```python
'data_dir': '/root/SteelClassification/Data',   # default value, change to your own path
```

For example, on Windows:

```python
'data_dir': r'D:\Users\20337\Desktop\MobileSteelNet-main (1)\MobileSteelNet-main\Data',
```

### 2. Start Training

```bash
cd Code
python train.py
```

On Windows PowerShell:

```powershell
cd Code; python train.py
```

### 3. Key Training Configuration

The default configuration in `main()` of `train.py` is as follows:

```python
config = {
    # Data configuration
    'data_dir': '/root/SteelClassification/Data',
    'dataset_type': 'NEU-DET',
    'batch_size': 16,
    'num_workers': 8,
    'image_size': 224,

    # Model configuration (MobileSteelNet only)
    'num_classes': 6,
    'in_channels': 3,

    # Training configuration
    'epochs': 100,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'scheduler': 'StepLR',
    'weight_decay': 1e-4,
    'data_ratio': 1.0,
    'train_split': 0.05,
    'val_split': 0.05,
    'test_split': 0.9,

    # Saving configuration
    'save_dir': 'checkpoints',
    'save_interval': 10,

    # Random seed
    'random_seed': 42,
}
```

Supported optimizers: `Adam`, `SGD`, `AdamW`. Supported learning-rate schedulers: `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`.

### 4. Train Programmatically

```python
from models import MobileSteelNet
from dataset import get_data_loaders, get_transforms
from train import SteelDefectTrainer

config = {
    'data_dir': r'path/to/Data',
    'dataset_type': 'NEU-DET',
    'batch_size': 16,
    'num_workers': 8,
    'image_size': 224,
    'num_classes': 6,
    'epochs': 100,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'scheduler': 'StepLR',
    'save_dir': 'checkpoints',
}

train_transform, val_transform = get_transforms(image_size=224, augmentation=True)
train_loader, val_loader, test_loader, class_names = get_data_loaders(
    data_dir=config['data_dir'],
    dataset_type=config['dataset_type'],
    batch_size=config['batch_size'],
    transform_train=train_transform,
    transform_val=val_transform,
)
config['class_names'] = class_names

trainer = SteelDefectTrainer(config)
trainer.train(train_loader, val_loader, test_loader)
```

---

## Network Architecture

The overall design follows the pattern "lightweight backbone + multi-scale fusion + efficient attention". The input is `3 × 224 × 224` and the output is a `num_classes`-dimensional classification vector.

```
Input (3 × 224 × 224)
   │
   ▼
Stem: Conv3×3, s2 + BN + ReLU                        → 32 × 112 × 112
   │
   ▼
Stage 1 (shallow features) 3 × DepthwiseSeparableConv → 128 × 56 × 56
   │
   ▼
Stage 2 (mid-level features) 4 × DepthwiseSeparableConv → 512 × 28 × 28
   │
   ▼
Stage 3 (deep features) 4 × DepthwiseSeparableConv    → 1024 × 14 × 14
   │
   ├──────────────► MSFF (multi-scale feature fusion of Stage1/2/3)
   │                        ↓
   │                 CIEMA (Cross-Interactive Efficient Multi-Scale Attention)
   │                        ↓
   └──────────────► Global Average Pooling → FC(1024 → num_classes)
```

### Stage-by-Stage Details

| Stage | Module | Output Channels | Output Size |
| --- | --- | --- | --- |
| Stem | `Conv2d(3→32, k=3, s=2, p=1)` + BN + ReLU | 32 | 112 × 112 |
| Stage 1 | `DWConv(32→64, s1)` → `DWConv(64→128, s2)` → `DWConv(128→128, s1)` | 128 | 56 × 56 |
| Stage 2 | `DWConv(128→256, s1)` → `DWConv(256→512, s2)` → `DWConv(512→512, s1)` ×2 | 512 | 28 × 28 |
| Stage 3 | `DWConv(512→512, s1)` ×2 → `DWConv(512→1024, s2)` → `DWConv(1024→1024, s1)` | 1024 | 14 × 14 |
| MSFF | Spatial resize → channel concatenation (1664) → `Conv1×1` + BN + ReLU | 1024 | 56 × 56 |
| CIEMA | Inter-channel interaction + coordinate attention + cross-spatial learning + final fusion | 1024 | 56 × 56 |
| Classifier | `AdaptiveAvgPool2d(1)` → `FC(1024 → num_classes)` | num_classes | — |

Here `DepthwiseSeparableConv` = 3×3 depthwise convolution (`groups=in_channels`) + 1×1 pointwise convolution + BatchNorm, used to greatly reduce computation and parameters while preserving representational capacity.

### Model Scale

| Metric | Value |
| --- | --- |
| Total / trainable parameters | 8,438,535 |
| Model size (as reported in the paper, INT8) | 8.2 MB |
| Model size (FP32 weights) | ≈ 32.19 MB |
| NEU-DET average accuracy | 91.36% |
| Scratch-class accuracy | 93.70% |

---

## The Two Key Innovations

### Innovation 1: MSFF — Multi-Scale Feature Fusion

**Problem it addresses**

The scale of steel surface defects varies greatly. Scratches and crazing are thin, elongated, low-contrast small targets whose discriminative information lies mainly in the high-resolution texture and edge features of shallow layers, whereas patches and rolled-in scale rely more on deep semantic features. If only the deep features from the last backbone layer are used, the details of small defects are largely lost during downsampling.

**How it works**

MSFF takes the feature maps from Stage 1 / Stage 2 / Stage 3 and fuses them in three steps:

1. **Spatial alignment**: The largest spatial resolution among the three stages (56×56 from Stage 1) is used as the target; the remaining feature maps are upsampled with bilinear interpolation so that all feature maps share the same size.
2. **Channel concatenation**: Concatenate along the channel dimension to obtain a multi-scale feature with `128 + 512 + 1024 = 1664` channels.
3. **Feature fusion**: Compress and fuse the 1664 channels into a unified 1024-channel output through `1×1 Conv → BN → ReLU`.

```python
class MSFF(nn.Module):
    def __init__(self, stage_channels, out_channels=None):
        super(MSFF, self).__init__()
        self.stage_channels = list(stage_channels)
        if out_channels is None:
            out_channels = max(self.stage_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(self.stage_channels), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
```

**Effect**

Fine-grained texture information from shallow layers and semantic information from deep layers are explicitly aggregated along the channel dimension, enabling the network to attend to both "fine" and "coarse" defects. In the paper, MobileSteelNet reaches **93.70%** accuracy on the Scratch class, an improvement of **82.12 percentage points** over the MobileNetV1 baseline.

---

### Innovation 2: CIEMA — Cross-Interactive Efficient Multi-Scale Attention

**Problem it addresses**

Existing attention mechanisms (such as SE, CBAM, ECA, and EMA) usually focus on a single dimension: either channel recalibration or spatial recalibration. Moreover, multi-scale spatial information is often extracted serially at a high computational cost, making it difficult to balance accuracy and efficiency in a lightweight network.

**How it works**

CIEMA unifies **inter-channel interaction**, **parallel multi-scale spatial extraction**, and **grouped efficient computation** within a single module, composed of the following sub-modules:

**(1) Inter-channel interaction — `ChannelInteraction`**

The input is processed by both global average pooling and global max pooling, then passed through a shared MLP (reduction ratio `reduction=16`) to produce channel attention weights. In parallel, a `1×1 Conv` generates channel-interaction features. The two are adaptively combined using the channel attention weights:

```
enhanced_x = x * channel_att + 1×1Conv(x) * (1 - channel_att)
```

**(2) Parallel multi-scale spatial extraction — `CrossSpatialLearning`**

Within each channel group, `1×1`, `3×3`, and `5×5` convolutions extract multi-scale spatial features in parallel. The results are concatenated into 3× channels, weighted by a spatial attention block (`1×1 Conv + BN + Sigmoid`), and then compressed back to the original per-group channel count with `1×1 Conv + BN`. Meanwhile, a `1×1 Conv` performs channel modeling, and a learnable scalar `alpha` weights the channel-modeling result before it is added to the multi-scale features:

```
enhanced_x = ms_reduce(SA(cat[f1, f3, f5])) + alpha * Conv1×1(x)
```

**(3) Grouped efficient computation and cross-branch interaction**

The features are split into `factor=32` groups along the channel dimension. Within each group, coordinate attention is applied (`pool_h` horizontal pooling + `pool_w` vertical pooling → `1×1 Conv` → split into H and W directional weights), producing a `1×1` branch and a `3×3` branch. The channel descriptors of the two branches then modulate each other (`softmax(agp(x1)) @ x2` and `softmax(agp(x2)) @ x1`), and the aggregated result forms a spatial weight that recalibrates the input, yielding the spatially enhanced feature.

**(4) Final fusion — `final_fusion`**

The multi-scale feature obtained from the channel-enhanced branch through a `1×1 → 1/4 → 1×1` bottleneck is concatenated with the spatially enhanced branch, and a `1×1 Conv + BN + Sigmoid` produces fusion weights that adaptively combine the two paths:

```
out = channel_enhanced_x * final_weights + spatial_enhanced * (1 - final_weights)
```

**Effect**

- **Accuracy**: 91.36% average accuracy on NEU-DET, surpassing ResNet-50 (88.01%) and MobileNetV2 (86.08%).
- **Efficiency**: The grouped and depthwise-separable computation keeps the model at only 8.2 MB, meeting the real-time requirements of edge deployment.

---

## Tensor Tests

`MobileSteelNet.py` includes complete tensor tests (shape validation + backward-pass validation), which can be run directly:

```bash
cd Code
python -m models.MobileSteelNet
```

Expected output:

```
[Test 1] CIEMA attention module
  Input  tensor shape: (2, 1024, 7, 7)
  Output tensor shape: (2, 1024, 7, 7)
  [OK] CIEMA input/output shape match
  [OK] CIEMA backward pass works

[Test 2] MSFF multi-scale feature fusion module
  Input stage1 shape: (2, 128, 56, 56)
  Input stage2 shape: (2, 512, 28, 28)
  Input stage3 shape: (2, 1024, 14, 14)
  Output shape: (2, 1024, 56, 56)
  [OK] MSFF multi-scale fusion shape is correct
  [OK] MSFF backward pass works

[Test 3] MobileSteelNet full network
  Input  tensor shape: (4, 3, 224, 224)
  Output tensor shape: (4, 6)
  [OK] Output shape is correct: (B, num_classes)
  Total parameters:     8,438,535
  Approx. 32.19 MB (FP32)
  [OK] Backward pass works, loss = ...

[Test 4] Robustness under different input sizes
  [OK] Input 160x160 -> Output (1, 6)
  [OK] Input 224x224 -> Output (1, 6)
  [OK] Input 256x256 -> Output (1, 6)

All tensor tests passed!
```

---

## Training Outputs

During training, models and results are automatically saved under `Code/checkpoints/`:

```
Code/checkpoints/
├── checkpoint_epoch_10.pth      # Periodically saved checkpoints (every 10 epochs by default)
├── checkpoint_epoch_20.pth
├── ...
├── best_model.pth               # Model with the best validation accuracy
└── results/                     # Evaluation results generated after training
    ├── metrics.txt              # Detailed metrics (overall accuracy, per-class precision/recall/F1)
    ├── training_history.json    # Training history (loss, accuracy, learning rate)
    ├── confusion_matrix.png     # Confusion matrix heatmap
    ├── training_curves.png      # Training/validation curves
    └── class_distribution.png   # Class distribution plot
```

Each checkpoint contains `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_val_accuracy`, `train_history`, and `config`, and training can be resumed via `trainer.load_checkpoint(path)`.

---

## Notes

1. Make sure you have enough GPU memory for training; reduce `batch_size` if you run out of memory.
2. `data_dir` must point to the **root directory** that contains `NEU-DET/` (or `NEU-DET.zip`), not to `NEU-DET/` itself.
3. Adjust `learning_rate` and `epochs` according to the size of your dataset.
4. Data augmentation is enabled by default (`get_transforms(..., augmentation=True)`), which helps improve generalization.
5. On Windows, if `num_workers` causes multiprocessing errors, set it to `0`.
6. To ensure reproducibility, `train.py` fixes the random seed (default `42`) and disables cuDNN benchmarking.
