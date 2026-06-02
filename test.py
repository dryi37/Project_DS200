import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# =============================================================================
# Transforms
# =============================================================================

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def make_tf(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

TF_SAM = make_tf(1024)   # sam2unet_conv / conv_bghr
TF_CNX = make_tf(448)    # sam2unet_conv / conv_bghr / bghr input
TF_352 = make_tf(352)    # sam2unet_bghr (trainsize mặc định)


# =============================================================================
# Model factory
# =============================================================================

def build_model(args, device):
    sys.path.insert(0, str(Path(__file__).parent / "models"))

    if args.model == "sam2unet_conv":
        from model import SAM2UNeXT
        model = SAM2UNeXT(sam2_checkpoint=args.sam2_ckpt, convnext_pretrained=False)

    elif args.model == "sam2unet_conv_bghr":
        from model_bg import SAM2UNeXT_BG
        model = SAM2UNeXT_BG(sam2_checkpoint=args.sam2_ckpt,
                             sam2_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
                             convnext_pretrained=False)

    elif args.model == "sam2unet_bghr":
        from SAM2UNet_BGHR import SAM2UNet_BGHR
        model = SAM2UNet_BGHR(checkpoint_path=args.sam2_ckpt)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd)

    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    dice  = ckpt.get("best_dice", "?") if isinstance(ckpt, dict) else "?"
    print(f"[Checkpoint] epoch={epoch}  best_dice={dice}")

    return model.to(device).eval()


# =============================================================================
# Inference: trả về prob map numpy (H, W) ở kích thước gốc orig_hw
# =============================================================================

@torch.no_grad()
def infer(model, img_pil, orig_hw, device, model_name, trainsize=352):
    if model_name == "sam2unet_conv":
        x_sam = TF_SAM(img_pil).unsqueeze(0).to(device)
        x_cnx = TF_CNX(img_pil).unsqueeze(0).to(device)
        logits = model(x_sam, x_cnx)                        # (1,1,352,352)

    elif model_name == "sam2unet_conv_bghr":
        x_sam = TF_SAM(img_pil).unsqueeze(0).to(device)
        x_cnx = TF_CNX(img_pil).unsqueeze(0).to(device)
        outputs = model(x_sam, x_cnx)
        logits = outputs["final"]                            # (1,1,352,352)

    else:  # sam2unet_bghr
        tf = make_tf(trainsize)
        x = tf(img_pil).unsqueeze(0).to(device)
        logits, *_ = model(x)                               # (1,1,trainsize,trainsize)

    # Resize về kích thước ảnh gốc để tính metric chính xác
    prob = torch.sigmoid(logits)
    prob = F.interpolate(prob, size=orig_hw, mode="bilinear", align_corners=False)
    return prob.squeeze().cpu().numpy().astype(np.float32)


# =============================================================================
# Metrics
# =============================================================================

def cal_metrics(pred, gt):
    pred_b = (pred >= 0.5).astype(np.float32)
    gt_b   = (gt   >= 0.5).astype(np.float32)
    inter  = (pred_b * gt_b).sum()
    dice   = (2 * inter + 1e-6) / (pred_b.sum() + gt_b.sum() + 1e-6)
    iou    = (inter + 1e-6) / (pred_b.sum() + gt_b.sum() - inter + 1e-6)
    mae    = float(np.abs(pred - gt).mean())
    return float(dice), float(iou), float(mae)


# =============================================================================
# Helpers
# =============================================================================

def find_dir(parent, candidates):
    for name in candidates:
        p = os.path.join(parent, name)
        if os.path.isdir(p):
            return p
    return None


def find_mask(mask_dir, stem):
    for ext in [".png", ".jpg", ".bmp", ".tif"]:
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


# =============================================================================
# Evaluate one dataset
# =============================================================================

def evaluate_dataset(model, img_dir, mask_dir, device, args):
    img_files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")))
    dices, ious, maes = [], [], []

    for fname in img_files:
        stem = os.path.splitext(fname)[0]
        mask_path = find_mask(mask_dir, stem)
        if mask_path is None:
            continue

        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        gt  = np.array(Image.open(mask_path).convert("L")) / 255.0
        orig_hw = gt.shape   # (H, W)

        prob = infer(model, img, orig_hw, device, args.model, args.trainsize)
        d, i, m = cal_metrics(prob, gt)
        dices.append(d); ious.append(i); maes.append(m)

    if len(dices) == 0:
        return None
    return {
        "dice": float(np.mean(dices)),
        "iou":  float(np.mean(ious)),
        "mae":  float(np.mean(maes)),
        "n":    len(dices),
    }


# =============================================================================
# Main
# =============================================================================

DATASETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model  : {args.model}")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    model = build_model(args, device)

    # Dùng danh sách dataset tùy chỉnh hoặc mặc định 5 dataset chuẩn
    datasets = args.datasets if args.datasets else DATASETS

    sep = "=" * 50
    all_results = {}

    for ds in datasets:
        ds_root = os.path.join(args.test_root, ds)
        if not os.path.isdir(ds_root):
            print(f"[Skip] {ds} — không tìm thấy {ds_root}")
            continue

        img_dir  = find_dir(ds_root, ["images", "image"])
        mask_dir = find_dir(ds_root, ["masks", "mask"])
        if not img_dir or not mask_dir:
            print(f"[Skip] {ds} — không có images/ hoặc masks/")
            continue

        print(f"\n{sep}\n  {ds}\n{sep}")
        r = evaluate_dataset(model, img_dir, mask_dir, device, args)
        if r is None:
            print("  [!] Không có ảnh hợp lệ")
            continue

        all_results[ds] = r
        print(f"  Samples : {r['n']}")
        print(f"  mDice   : {r['dice']:.4f}")
        print(f"  mIoU    : {r['iou']:.4f}")
        print(f"  MAE     : {r['mae']:.4f}")

    # Tổng kết
    if len(all_results) > 1:
        print(f"\n{sep}")
        print(f"  SUMMARY — {args.model}")
        print(sep)
        print(f"  {'Dataset':<26} {'Dice':>7} {'IoU':>7} {'MAE':>7}")
        print(f"  {'-'*26} {'-'*7} {'-'*7} {'-'*7}")
        for name, r in all_results.items():
            print(f"  {name:<26} {r['dice']:>7.4f} {r['iou']:>7.4f} {r['mae']:>7.4f}")
        avg_dice = np.mean([r["dice"] for r in all_results.values()])
        avg_iou  = np.mean([r["iou"]  for r in all_results.values()])
        avg_mae  = np.mean([r["mae"]  for r in all_results.values()])
        print(f"  {'Average':<26} {avg_dice:>7.4f} {avg_iou:>7.4f} {avg_mae:>7.4f}")
        print(sep)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified evaluation script")

    p.add_argument("--model", type=str, required=True,
                   choices=["sam2unet_conv", "sam2unet_conv_bghr", "sam2unet_bghr"])
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Model checkpoint (.pt)")
    p.add_argument("--test_root",  type=str, required=True,
                   help="Thư mục chứa các dataset con (Kvasir, CVC-ClinicDB, ...)")
    p.add_argument("--sam2_ckpt",  type=str, default=None,
                   help="SAM2 Hiera checkpoint (nếu không truyền thì dùng config mặc định)")
    p.add_argument("--datasets",   type=str, nargs="*", default=None,
                   help="Chỉ định dataset cụ thể, mặc định test cả 5")
    p.add_argument("--trainsize",  type=int, default=352,
                   help="[bghr] Input size (mặc định 352)")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())