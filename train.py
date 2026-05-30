import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **kw): self._it = iterable
        def __iter__(self): return iter(self._it)
        def set_postfix(self, **kw): pass
        def __len__(self): return len(self._it)
        def __next__(self): return next(iter(self._it))
        @property
        def n(self): return 0


# =============================================================================
# Loss functions
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(logits.size(0), -1)
        t = targets.view(targets.size(0), -1)
        inter = (p * t).sum(dim=1)
        dice = (2 * inter + self.smooth) / (p.sum(dim=1) + t.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, w_bce=0.5, w_dice=0.5):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.w_bce * self.bce(logits, targets) + self.w_dice * self.dice(logits, targets)


def structure_loss(pred, mask):
    """Weighted BCE + weighted IoU — dùng cho sam2unet_bghr."""
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_prob = torch.sigmoid(pred)
    inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
    union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def mask_to_boundary(mask, k=5):
    mask = (mask > 0.5).float()
    pad = k // 2
    dilated = F.max_pool2d(mask, k, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, k, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


def soft_boundary_from_logits(logit, k=5):
    prob = torch.sigmoid(logit)
    pad = k // 2
    dilated = F.max_pool2d(prob, k, stride=1, padding=pad)
    eroded = -F.max_pool2d(-prob, k, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


def dice_loss_prob(prob, target, smooth=1.0):
    inter = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (2.0 * inter + smooth) / (denom + smooth)).mean()


def edge_loss_fn(pred_edge_logit, gt_mask_or_edge, is_edge_map=False):
    """
    Dùng cho cả 2 kiểu:
      - sam2unet_conv_bghr: gt là mask thô, tự tính edge
      - sam2unet_bghr:      gt đã là edge map
    """
    gt_edge = gt_mask_or_edge if is_edge_map else mask_to_boundary(gt_mask_or_edge)
    if is_edge_map:
        # weighted BCE + Dice (sam2unet_bghr style)
        pos = gt_edge.sum()
        neg = gt_edge.numel() - pos
        pos_weight = torch.clamp((neg / (pos + 1.0)).detach(), 1.0, 20.0).reshape(1)
        bce = F.binary_cross_entropy_with_logits(pred_edge_logit, gt_edge, pos_weight=pos_weight)
        dice = dice_loss_prob(torch.sigmoid(pred_edge_logit), gt_edge)
        return bce + dice
    else:
        # Dice only (sam2unet_conv_bghr style)
        p = torch.sigmoid(pred_edge_logit)
        inter = (p * gt_edge).sum(dim=(2, 3))
        denom = p.sum(dim=(2, 3)) + gt_edge.sum(dim=(2, 3))
        return (1 - (2 * inter + 1) / (denom + 1)).mean()


def boundary_loss_fn(pred_mask_logit, gt_mask):
    gt_b = mask_to_boundary(gt_mask)
    pred_b = soft_boundary_from_logits(pred_mask_logit)
    return dice_loss_prob(pred_b, gt_b)


# =============================================================================
# Metrics
# =============================================================================

@torch.no_grad()
def dice_score(logits, targets, thr=0.5, smooth=1.0):
    p = (torch.sigmoid(logits) > thr).float().view(logits.size(0), -1)
    t = targets.view(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    return ((2 * inter + smooth) / (p.sum(dim=1) + t.sum(dim=1) + smooth)).mean().item()


# =============================================================================
# Per-model: forward + loss
# =============================================================================

_BCE_DICE = BCEDiceLoss(0.5, 0.5)

# --- sam2unet_conv ---
def forward_loss_conv(model, batch, device):
    img_sam = batch["img_sam"].to(device, non_blocking=True)
    img_cnx = batch["img_cnx"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    logits = model(img_sam, img_cnx)          # (B,1,H,W)
    loss = _BCE_DICE(logits, mask)
    return logits, loss


# --- sam2unet_conv_bghr ---
def forward_loss_conv_bghr(model, batch, device, w):
    x_sam = batch["img_sam"].to(device, non_blocking=True)
    x_cnx = batch["img_cnx"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    outputs = model(x_sam, x_cnx)             # dict: final, coarse, s1, s2, edge
    l_final  = _BCE_DICE(outputs["final"],  mask)
    l_coarse = _BCE_DICE(outputs["coarse"], mask)
    l_s1     = _BCE_DICE(outputs["s1"],     mask)
    l_s2     = _BCE_DICE(outputs["s2"],     mask)
    l_edge   = edge_loss_fn(outputs["edge"], mask, is_edge_map=False)
    l_bd     = boundary_loss_fn(outputs["final"], mask)
    loss = (w["final"] * l_final + w["coarse"] * l_coarse
            + w["s1"] * l_s1 + w["s2"] * l_s2
            + w["edge"] * l_edge + w["boundary"] * l_bd)
    return outputs["final"], loss


# --- sam2unet_bghr ---
def maybe_multiscale_resize(x, target, base_size):
    scale = random.choice([1.0, 1.25])
    size = int(round(base_size * scale / 32) * 32)
    if size == x.shape[-1]:
        return x, target
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    target = F.interpolate(target, size=(size, size), mode="nearest")
    return x, target


def forward_loss_bghr(model, batch, device, args):
    # SAM2UNet_BGHR output size = input size, mask là 352x352
    x = batch["img_cnx"].to(device, non_blocking=True)
    target = batch["mask"].to(device, non_blocking=True)
    target = (target > 0.5).float()
    # Resize input về trainsize (mặc định 352) để output khớp với mask
    x = F.interpolate(x, size=(args.trainsize, args.trainsize),
                      mode="bilinear", align_corners=False)
    if args.multi_scale:
        x, target = maybe_multiscale_resize(x, target, args.trainsize)
    # returns: pred_final, pred_side1, pred_side2, pred_edge, pred_coarse
    pred_final, pred_side1, pred_side2, pred_edge, pred_coarse = model(x)
    gt_edge = mask_to_boundary(target, k=5)
    loss_final  = structure_loss(pred_final,  target)
    loss_side1  = structure_loss(pred_side1,  target)
    loss_side2  = structure_loss(pred_side2,  target)
    loss_coarse = structure_loss(pred_coarse, target)
    loss_e = edge_loss_fn(pred_edge, gt_edge, is_edge_map=True)
    loss_b = boundary_loss_fn(pred_final, target)
    loss = (1.50 * loss_final + 0.60 * loss_side1 + 0.90 * loss_side2
            + args.lambda_coarse * loss_coarse
            + args.lambda_edge * loss_e
            + args.lambda_boundary * loss_b)
    return pred_final, loss


# =============================================================================
# Model + DataLoader factory
# =============================================================================

def build_model_and_loaders(args, device):
    """
    Trả về (model, train_loader, val_loader) cho cả 3 model.
    """
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent))
    from dataset import get_loader
    m = args.model

    # ---- sam2unet_conv ----
    if m == "sam2unet_conv":
        import sys; sys.path.insert(0, str(Path(__file__).parent / "models"))
        from model import SAM2UNeXT
        model = SAM2UNeXT(sam2_checkpoint=args.sam2_ckpt, convnext_pretrained=True).to(device)
        train_loader = get_loader(args.train_dir, batch_size=args.batch_size,
                                  mode="train", num_workers=args.num_workers)
        val_loader = get_loader(args.val_dir, batch_size=1,
                                mode="val", num_workers=args.num_workers)
        return model, train_loader, val_loader

    # ---- sam2unet_conv_bghr ----
    if m == "sam2unet_conv_bghr":
        import sys; sys.path.insert(0, str(Path(__file__).parent / "models"))
        from model_bg import SAM2UNeXT_BG, load_friend_checkpoint
        model = SAM2UNeXT_BG(sam2_checkpoint=args.sam2_ckpt, convnext_pretrained=True).to(device)
        if args.init_from and Path(args.init_from).exists():
            load_friend_checkpoint(model, args.init_from)
            print(f"[init_from] Loaded weights from {args.init_from}")
        train_loader = get_loader(args.train_dir, batch_size=args.batch_size,
                                  mode="train", num_workers=args.num_workers)
        val_loader = get_loader(args.val_dir, batch_size=1,
                                mode="val", num_workers=args.num_workers)
        return model, train_loader, val_loader

    # ---- sam2unet_bghr ----
    if m == "sam2unet_bghr":
        import sys; sys.path.insert(0, str(Path(__file__).parent / "models"))
        from SAM2UNet_BGHR import SAM2UNet_BGHR

        # seed
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        model = SAM2UNet_BGHR(args.sam2_ckpt).to(device)

        if args.resume and Path(args.resume).exists():
            _load_bghr_checkpoint(model, args.resume, device)

        train_loader = get_loader(args.train_dir, batch_size=args.batch_size,
                                  mode="train", num_workers=args.num_workers)
        val_loader   = get_loader(args.val_dir, batch_size=1,
                                  mode="val", num_workers=args.num_workers)
        return model, train_loader, val_loader

    raise ValueError(f"Unknown model: {args.model}")


def _load_bghr_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if ckpt and next(iter(ckpt)).startswith("module."):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    msg = model.load_state_dict(ckpt, strict=False)
    print(f"[Resume] {ckpt_path}  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")


# =============================================================================
# Training / Validation loops
# =============================================================================

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args):
    model.train()
    total_loss = 0.0
    w = {"final": 1.0, "coarse": 0.5, "s1": 0.3, "s2": 0.3, "edge": 0.3, "boundary": 0.3}

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for i, batch in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            if args.model == "sam2unet_conv":
                logits, loss = forward_loss_conv(model, batch, device)
            elif args.model == "sam2unet_conv_bghr":
                logits, loss = forward_loss_conv_bghr(model, batch, device, w)
            else:  # sam2unet_bghr
                logits, loss = forward_loss_bghr(model, batch, device, args)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if args.model == "sam2unet_bghr" and i % 50 == 0:
            print(f"  iter {i+1}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, epoch, args):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    w = {"final": 1.0, "coarse": 0.5, "s1": 0.3, "s2": 0.3, "edge": 0.3, "boundary": 0.3}

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [ val ]", leave=False)
    for i, batch in enumerate(pbar):
        with autocast():
            if args.model == "sam2unet_conv":
                logits, loss = forward_loss_conv(model, batch, device)
                gt = batch["mask"].to(device, non_blocking=True)
            elif args.model == "sam2unet_conv_bghr":
                logits, loss = forward_loss_conv_bghr(model, batch, device, w)
                gt = batch["mask"].to(device, non_blocking=True)
            else:  # sam2unet_bghr — dùng chung PolypDataset, key "mask"
                logits, loss = forward_loss_bghr(model, batch, device, args)
                gt = batch["mask"].to(device, non_blocking=True)
        total_loss += loss.item()
        total_dice += dice_score(logits, gt)
        pbar.set_postfix(dice=f"{total_dice/(i+1):.4f}")

    n = len(loader)
    return total_loss / n, total_dice / n


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_dice, val_dice=None):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimiser": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_dice": best_dice,
        "val_dice": val_dice,
    }, path)


def resume_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimiser"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"[Resume] {path}  epoch={ckpt['epoch']}  best_dice={ckpt['best_dice']:.4f}")
    return ckpt["epoch"] + 1, ckpt["best_dice"]


# =============================================================================
# Main
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model : {args.model}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    model, train_loader, val_loader = build_model_and_loaders(args, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    scaler = GradScaler()

    save_dir = Path(args.save_dir) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_dice = 1, 0.0
    if args.resume and Path(args.resume).exists() and args.model != "sam2unet_bghr":
        start_epoch, best_dice = resume_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device)

    print("\n" + "=" * 60 + f"\nStarting training — {args.model}\n" + "=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        # ----- val -----
        val_loss, val_dice = validate(model, val_loader, device, epoch, args)
        print(f"Epoch {epoch:03d}/{args.epochs:03d}  "
              f"| train={train_loss:.4f} | val={val_loss:.4f} "
              f"| dice={val_dice:.4f} | lr={lr_now:.2e} | {elapsed:.0f}s")
        if device.type == "cuda":
            print(f"          VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
        save_checkpoint(save_dir / "last.pt", epoch, model, optimizer,
                        scheduler, scaler, best_dice, val_dice)
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(save_dir / "best.pt", epoch, model, optimizer,
                            scheduler, scaler, best_dice, val_dice)
            print(f"          ★ New best Dice: {best_dice:.4f}")

    print(f"\nDone. Best val Dice = {best_dice:.4f}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified training script")

    # Model selector
    p.add_argument("--model", type=str, required=True,
                   choices=["sam2unet_conv", "sam2unet_conv_bghr", "sam2unet_bghr"],
                   help="Which model variant to train")

    # --- Paths (dùng chung cho cả 3 model) ---
    p.add_argument("--train_dir", type=str, required=True,
                   help="Training root (phải có images/ và masks/ bên trong)")
    p.add_argument("--val_dir",   type=str, required=True,
                   help="Validation root (phải có images/ và masks/ bên trong)")

    # --- Shared paths ---
    p.add_argument("--sam2_ckpt",  type=str, default=None,         help="SAM2 Hiera checkpoint")
    p.add_argument("--save_dir",   type=str, default="checkpoints", help="Root save directory")
    p.add_argument("--resume",     type=str, default=None,          help="Resume checkpoint path")

    # --- conv_bghr only ---
    p.add_argument("--init_from",  type=str, default=None, help="[conv_bghr] Friend checkpoint")

    # --- Hyperparams ---
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=4)

    # --- bghr only ---
    p.add_argument("--trainsize",        type=int,   default=352)
    p.add_argument("--amp",              action="store_true")
    p.add_argument("--multi_scale",      action="store_true")
    p.add_argument("--seed",             type=int,   default=1024)
    p.add_argument("--lambda_edge",      type=float, default=0.30)
    p.add_argument("--lambda_boundary",  type=float, default=0.50)
    p.add_argument("--lambda_coarse",    type=float, default=0.50)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)