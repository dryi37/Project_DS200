import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.amp import autocast

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **kw): self._it = iterable
        def __iter__(self): return iter(self._it)


# =============================================================================
# Metrics
# =============================================================================

def _dice(pred_bin: np.ndarray, gt_bin: np.ndarray, smooth: float = 1.0) -> float:
    inter = (pred_bin * gt_bin).sum()
    return (2.0 * inter + smooth) / (pred_bin.sum() + gt_bin.sum() + smooth)


def _iou(pred_bin: np.ndarray, gt_bin: np.ndarray, smooth: float = 1.0) -> float:
    inter = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - inter
    return (inter + smooth) / (union + smooth)


def _mae(pred_prob: np.ndarray, gt_bin: np.ndarray) -> float:
    return float(np.abs(pred_prob - gt_bin).mean())


# =============================================================================
# Visualisation
# =============================================================================

def save_visualisation(img_rgb, gt_mask, pred_mask, save_path, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_rgb);       axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(gt_mask,  cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth");  axes[1].axis("off")
    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction");    axes[2].axis("off")
    if title:
        fig.suptitle(title, fontsize=9, y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


# =============================================================================
# Model factory
# =============================================================================

def build_model(args, device):
    root = Path(__file__).parent
    sys.path.insert(0, str(root / "models"))

    if args.model == "sam2unet_conv":
        from model import SAM2UNeXT
        model = SAM2UNeXT(sam2_checkpoint=args.sam2_ckpt, convnext_pretrained=False)

    elif args.model == "sam2unet_conv_bghr":
        from model_bg import SAM2UNeXT_BG
        model = SAM2UNeXT_BG(sam2_checkpoint=args.sam2_ckpt, convnext_pretrained=False)

    elif args.model == "sam2unet_bghr":
        from SAM2UNet_BGHR import SAM2UNet_BGHR
        model = SAM2UNet_BGHR(checkpoint_path=args.sam2_ckpt)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    # sam2unet_bghr lưu state_dict trực tiếp, 2 model kia lưu dict có key "model"
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)

    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    dice  = ckpt.get("best_dice", "?") if isinstance(ckpt, dict) else "?"
    print(f"[Checkpoint] {args.ckpt}  epoch={epoch}  best_dice={dice}")

    return model.to(device)


# =============================================================================
# Inference: lấy logit final từ mỗi model
# =============================================================================

def get_logits(model, batch, device, args):
    """Trả về logit (B,1,H,W) và gt mask (B,1,H,W)."""
    if args.model == "sam2unet_conv":
        img_sam = batch["img_sam"].to(device, non_blocking=True)
        img_cnx = batch["img_cnx"].to(device, non_blocking=True)
        logits  = model(img_sam, img_cnx)                        # (B,1,352,352)

    elif args.model == "sam2unet_conv_bghr":
        img_sam = batch["img_sam"].to(device, non_blocking=True)
        img_cnx = batch["img_cnx"].to(device, non_blocking=True)
        outputs = model(img_sam, img_cnx)                        # dict
        logits  = outputs["final"]                               # (B,1,352,352)

    else:  # sam2unet_bghr — chỉ dùng img_cnx resize về trainsize
        img_cnx = batch["img_cnx"].to(device, non_blocking=True)
        x = F.interpolate(img_cnx, size=(args.trainsize, args.trainsize),
                          mode="bilinear", align_corners=False)
        logits, *_ = model(x)                                    # (B,1,trainsize,trainsize)
        # Resize về 352 để khớp mask
        logits = F.interpolate(logits, size=(352, 352), mode="bilinear", align_corners=False)

    mask = batch["mask"].to(device, non_blocking=True)           # (B,1,352,352)
    return logits, mask


# =============================================================================
# Evaluation loop
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, device, args, save_vis=False, vis_dir=None):
    model.eval()
    dice_list, iou_list, mae_list = [], [], []

    pbar = tqdm(loader, desc="Evaluating")
    for batch in pbar:
        stems = batch["stem"]

        with autocast("cuda"):
            logits, masks = get_logits(model, batch, device, args)

        probs = torch.sigmoid(logits)

        for b in range(probs.size(0)):
            prob_np  = probs[b, 0].cpu().numpy().astype(np.float32)
            gt_np    = masks[b, 0].cpu().numpy().astype(np.float32)
            pred_bin = (prob_np > args.threshold).astype(np.float32)

            dice_list.append(_dice(pred_bin, gt_np))
            iou_list.append(_iou(pred_bin, gt_np))
            mae_list.append(_mae(prob_np, gt_np))

            if save_vis and vis_dir is not None:
                _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_t = batch["img_sam"][b].cpu().permute(1, 2, 0).numpy()
                img_display = np.clip(img_t * _std + _mean, 0, 1)
                img_display = (img_display * 255).astype(np.uint8)
                img_display = cv2.resize(img_display, (352, 352))
                save_visualisation(
                    img_rgb=img_display, gt_mask=gt_np, pred_mask=pred_bin,
                    save_path=vis_dir / f"{stems[b]}.png",
                    title=(f"{stems[b]} | Dice={dice_list[-1]:.3f} "
                           f"IoU={iou_list[-1]:.3f} MAE={mae_list[-1]:.4f}"),
                )

    return {
        "dice": float(np.mean(dice_list)),
        "iou":  float(np.mean(iou_list)),
        "mae":  float(np.mean(mae_list)),
        "n":    len(dice_list),
    }


# =============================================================================
# Main
# =============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model  : {args.model}")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # Dataset import từ root
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import get_loader

    model = build_model(args, device)
    model.eval()

    sep = "=" * 55
    all_results = {}

    for test_dir in args.test_dir:
        dataset_name = Path(test_dir).name
        print(f"\n{sep}")
        print(f"  Dataset : {dataset_name}  ({test_dir})")
        print(sep)

        loader = get_loader(test_dir, batch_size=1, mode="val", num_workers=args.num_workers)

        vis_dir = None
        if args.save_vis:
            vis_dir = Path(args.vis_dir) / args.model / dataset_name
            vis_dir.mkdir(parents=True, exist_ok=True)

        results = evaluate(model, loader, device, args,
                           save_vis=args.save_vis, vis_dir=vis_dir)
        all_results[dataset_name] = results

        print(f"  Samples         : {results['n']}")
        print(f"  Mean Dice Score : {results['dice']:.4f}")
        print(f"  Mean IoU        : {results['iou']:.4f}")
        print(f"  Mean MAE        : {results['mae']:.4f}")
        if args.save_vis:
            print(f"  Visualisations  : {vis_dir}")

    # Tổng kết nếu test nhiều dataset
    if len(all_results) > 1:
        print(f"\n{sep}")
        print(f"  SUMMARY — {args.model}")
        print(sep)
        print(f"  {'Dataset':<22} {'Dice':>8} {'IoU':>8} {'MAE':>8}")
        print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
        for name, r in all_results.items():
            print(f"  {name:<22} {r['dice']:>8.4f} {r['iou']:>8.4f} {r['mae']:>8.4f}")
        avg_dice = np.mean([r["dice"] for r in all_results.values()])
        avg_iou  = np.mean([r["iou"]  for r in all_results.values()])
        avg_mae  = np.mean([r["mae"]  for r in all_results.values()])
        print(f"  {'Average':<22} {avg_dice:>8.4f} {avg_iou:>8.4f} {avg_mae:>8.4f}")
    print(sep)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified evaluation script")

    p.add_argument("--model", type=str, required=True,
                   choices=["sam2unet_conv", "sam2unet_conv_bghr", "sam2unet_bghr"],
                   help="Model variant")
    p.add_argument("--test_dir", type=str, nargs="+", required=True,
                   help="Một hoặc nhiều test dataset root (mỗi cái có images/ và masks/)")
    p.add_argument("--ckpt",      type=str, required=True, help="Model checkpoint (.pt)")
    p.add_argument("--sam2_ckpt", type=str, default=None,  help="SAM2 Hiera checkpoint")

    p.add_argument("--vis_dir",    type=str,   default="results/vis", help="Root thư mục lưu ảnh")
    p.add_argument("--save_vis",   action="store_true",               help="Lưu ảnh visualisation")
    p.add_argument("--threshold",  type=float, default=0.5,           help="Ngưỡng binarise")
    p.add_argument("--num_workers", type=int,  default=2)

    # bghr only
    p.add_argument("--trainsize", type=int, default=352,
                   help="[bghr] Resize input về size này trước khi đưa vào model")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())