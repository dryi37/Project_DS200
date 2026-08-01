# SAM2-DEB-UNet: Polyp Segmentation with SAM2 and Boundary-Guided Refinement

PyTorch implementation of a dual-encoder polyp segmentation framework built upon a frozen SAM2 Hiera-L backbone and a ConvNeXt-V2 Tiny auxiliary encoder. The proposed architecture incorporates Boundary-Guided High-Resolution (BGHR) refinement to improve boundary localization and segmentation accuracy on challenging polyp datasets.

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="1000">
</p>

---

## Installation

### Clone Repository

```bash
git clone https://github.com/dryi37/SAM2-DEB-UNet.git
cd SAM2-DEB-UNet
```

### Create Virtual Environment (Optional)

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
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

---

## Training

```bash
python train.py \
  --model sam2unet_conv_bghr \
  --train_dir data/TrainDataset \
  --val_dir data/TestDataset/Kvasir \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt \
  --init_from checkpoints/sam2unet_conv/best.pt \
  --epochs 50 \
  --batch_size 12 \
  --lr 1e-4
```

---

## Experiment Tracking (MLflow)

This project supports experiment tracking with MLflow.

### Start MLflow Server

```bash
docker compose up -d
```

Open:

```
http://localhost:5000
```

During training, the following information is automatically logged:

- Hyperparameters
- Training / Validation Loss
- Validation Dice Score
- Learning Rate
- GPU Memory Usage
- Best and Last Checkpoints
- Trained PyTorch Model

---

## Evaluation

### Evaluate on All Datasets

```bash
python test.py \
  --model sam2unet_bghr \
  --checkpoint checkpoints/sam2unet_conv_bghr/best.pt \
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

## Results

### Ablation Study

<p align="center">
  <img src="assets/ablation_study.png" width="800">
</p>

### Comparison with State-of-the-Art Methods

<p align="center">
  <img src="assets/sota_comparison.png" width="900">
</p>

Our method achieves the best average performance (0.873 mDice / 0.812 mIoU) and demonstrates consistent improvements across challenging datasets such as ColonDB and ETIS.

## Deployment

The trained model can be exported to ONNX and deployed with ONNX Runtime, FastAPI, and Docker for efficient production inference.

### Export to ONNX

```bash
cd deploy

python export_onnx.py \
    --checkpoint ../checkpoints/sam2unet_conv_bghr/best.pt \
    --sam2_ckpt ../checkpoints/sam2.1_hiera_large.pt \
    --output sam2unet_conv_bghr.onnx \
    --opset 19
```

The export script automatically verifies numerical consistency between the PyTorch and ONNX models to ensure the exported model produces consistent predictions.

### Run the API Server

```bash
cd deploy

pip install -r requirements-deploy.txt

uvicorn app:app --host 0.0.0.0 --port 8000
```

The API automatically uses the CUDA Execution Provider when available and falls back to CPU otherwise.

### Docker

Build the Docker image:

```bash
cd deploy

docker build -t sam2-polyp-api .
```

Run the container:

```bash
docker run -p 8000:8000 sam2-polyp-api
```

### API Endpoints

| Method | Endpoint | Description |
| :----: | :------: | ----------- |
| GET | `/health` | Check API status and execution provider |
| POST | `/predict` | Return the predicted segmentation mask as a PNG image |
| POST | `/predict/json` | Return the segmentation mask (Base64) together with inference statistics |

---

## Acknowledgements

```bibtex
@article{xiong2026sam2,
  title={Sam2-unet: Segment anything 2 makes strong encoder for natural and medical image segmentation},
  author={Xiong, Xinyu and Wu, Zihuang and Tan, Shuangyi and Li, Wenxue and Tang, Feilong and Chen, Ying and Li, Siying and Ma, Jie and Li, Guanbin},
  journal={Visual Intelligence},
  volume={4},
  number={1},
  pages={2},
  year={2026},
  publisher={Springer}
}
```

