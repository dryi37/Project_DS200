# SAM2-DEB-UNet: Polyp Segmentation with SAM2 and Boundary-Guided Refinement

PyTorch implementation of polyp segmentation models based on a frozen SAM2 Hiera-L image encoder, ConvNeXt-V2 Tiny auxiliary encoder, and Boundary-Guided High-Resolution (BGHR) refinement modules.

The repository provides three model variants for benchmarking the impact of dual-encoder feature fusion and boundary-aware supervision on gastrointestinal polyp segmentation.

---

## Overview

### Baseline Architecture

* Frozen SAM2 Hiera-L image encoder
* ConvNeXt-V2 Tiny auxiliary encoder
* U-Net style decoder
* BCE + Dice loss

### BGHR Enhancement

Additional components:

* Boundary-Guided High-Resolution refinement
* Edge prediction branch
* Deep supervision
* Boundary-aware optimization

---

## Model Variants

| Model                | Description                                     | Loss Function   |
| -------------------- | ----------------------------------------------- | --------------- |
| `sam2unet_conv`      | SAM2 + ConvNeXt dual encoder with U-Net decoder | BCE + Dice      |
| `sam2unet_conv_bghr` | Dual encoder + BGHR refinement                  | Multi-task loss |
| `sam2unet_bghr`      | Standalone BGHR architecture with RFB modules   | Structure loss  |

---

## Ablation Study (mDice)

| Dual Encoder | BGHR | Kvasir    | ClinicDB  | ColonDB   | CVC-300   | ETIS      | Avg       |
| ------------ | ---- | --------- | --------- | --------- | --------- | --------- | --------- |
|              |      | 0.915     | 0.888     | 0.800     | 0.887     | 0.782     | 0.854     |
|              |  ✓   | 0.923     | 0.905     | 0.790     | 0.893     | 0.806     | 0.863     |
| ✓            | ✓    | **0.935** | **0.916** | **0.811** | **0.876** | **0.828** | **0.873** |

The results demonstrate that combining dual-encoder features with BGHR refinement consistently improves segmentation performance across all benchmark datasets.

---

## Installation

### Clone Repository

```bash
git clone <your-repository-url>
cd SAM2_Polyp
```

### Install Dependencies

```bash
pip install torch torchvision
pip install timm albumentations opencv-python numpy pillow tqdm matplotlib gdown
```

### Install SAM2

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

---

## Download SAM2 Checkpoint

Create checkpoint directory:

```bash
mkdir -p checkpoints
```

Download SAM2 Hiera-L checkpoint:

```bash
wget -P checkpoints/ \
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Expected structure:

```text
checkpoints/
└── sam2.1_hiera_large.pt
```

---

## Dataset Preparation

The project uses the standard PraNet train/test split.

### Create Data Directory

```bash
mkdir -p data
cd data
```

### Download Training Dataset

```bash
gdown --fuzzy \
"https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing" \
-O TrainDataset.zip
```

Contains:

* Kvasir-SEG
* CVC-ClinicDB

### Download Test Dataset

```bash
gdown --fuzzy \
"https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing" \
-O TestDataset.zip
```

Contains:

* Kvasir
* CVC-ClinicDB
* CVC-ColonDB
* CVC-300
* ETIS-LaribPolypDB

### Extract

```bash
unzip TrainDataset.zip
unzip TestDataset.zip
```

Optional:

```bash
rm TrainDataset.zip TestDataset.zip
```

Final structure:

```text
data/
├── TrainDataset/
│   ├── images/
│   └── masks/
│
└── TestDataset/
    ├── Kvasir/
    │   ├── images/
    │   └── masks/
    │
    ├── CVC-ClinicDB/
    ├── CVC-ColonDB/
    ├── CVC-300/
    └── ETIS-LaribPolypDB/
```

---

## Training

### 1. Baseline Model

```bash
python train.py \
  --model sam2unet_conv \
  --train_dir data/TrainDataset \
  --val_dir data/TestDataset/Kvasir \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4
```

### 2. Dual Encoder + BGHR

```bash
python train.py \
  --model sam2unet_conv_bghr \
  --train_dir data/TrainDataset \
  --val_dir data/TestDataset/Kvasir \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt \
  --init_from checkpoints/sam2unet_conv/best.pt \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4
```

### 3. BGHR Model

```bash
python train.py \
  --model sam2unet_bghr \
  --train_dir data/TrainDataset \
  --val_dir data/TestDataset/Kvasir \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4 \
  --multi_scale \
  --trainsize 352
```

---

## Checkpoints

Checkpoints are automatically saved during training:

```text
checkpoints/
├── sam2.1_hiera_large.pt
│
├── sam2unet_conv/
│   ├── best.pt
│   └── last.pt
│
├── sam2unet_conv_bghr/
│   ├── best.pt
│   └── last.pt
│
└── sam2unet_bghr/
    ├── best.pt
    └── last.pt
```

Where:

* `best.pt` = highest validation Dice score
* `last.pt` = latest training checkpoint

---

## Evaluation

### Evaluate on All Datasets

```bash
python test.py \
  --model sam2unet_bghr \
  --checkpoint checkpoints/sam2unet_bghr/best.pt \
  --test_root data/TestDataset \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt
```

### Evaluate Specific Datasets

```bash
python test.py \
  --model sam2unet_conv_bghr \
  --checkpoint checkpoints/sam2unet_conv_bghr/best.pt \
  --test_root data/TestDataset \
  --datasets Kvasir CVC-ClinicDB
```

---

## Evaluation Metrics

The testing script reports:

* mDice
* mIoU
* MAE

for each dataset and an overall average.

Example output:

```text
Dataset              Dice     IoU      MAE
---------------------------------------------
Kvasir               0.935    0.901    0.021
CVC-ClinicDB         0.916    0.876    0.018
CVC-ColonDB          0.811    0.742    0.038
CVC-300              0.876    0.823    0.027
ETIS                 0.828    0.761    0.044
---------------------------------------------
Average              0.873    0.821    0.030
```

---

## Project Structure

```text
SAM2_Polyp/
│
├── train.py
├── test.py
├── README.md
│
├── models/
│   ├── sam2unet_conv.py
│   ├── sam2unet_conv_bghr.py
│   └── sam2unet_bghr.py
│
├── datasets/
├── checkpoints/
└── data/
```

---

## License

This project is intended for research and educational purposes.
