import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).parent.parent / "SAM2-DEB-UNet" / "models"
sys.path.insert(0, str(MODELS_DIR))


class InferenceWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x_sam: torch.Tensor, x_cnx: torch.Tensor) -> torch.Tensor:
        out = self.model(x_sam, x_cnx)
        return torch.sigmoid(out["final"])


def load_model(checkpoint: str, sam2_ckpt: str | None, device: torch.device) -> nn.Module:
    from model_bg import SAM2UNeXT_BG  # noqa: E402  (path injected above)

    model = SAM2UNeXT_BG(
        sam2_checkpoint=sam2_ckpt,
        sam2_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        convnext_pretrained=False,  # weights come from the checkpoint anyway
    )
    ckpt = torch.load(checkpoint, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd)
    return model.to(device).eval()


def export(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export_onnx] device = {device}")

    model = load_model(args.checkpoint, args.sam2_ckpt, device)
    wrapped = InferenceWrapper(model).to(device).eval()

    x_sam = torch.randn(1, 3, 1024, 1024, device=device)
    x_cnx = torch.randn(1, 3, 448, 448, device=device)

    with torch.no_grad():
        torch_out = wrapped(x_sam, x_cnx).cpu().numpy()

    print(f"[export_onnx] Exporting to {args.output} (opset {args.opset}) ...")
    torch.onnx.export(
        wrapped,
        (x_sam, x_cnx),
        args.output,
        input_names=["x_sam", "x_cnx"],
        output_names=["mask_prob"],
        opset_version=args.opset,
        dynamic_axes={
            "x_sam": {0: "batch"},
            "x_cnx": {0: "batch"},
            "mask_prob": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print("[export_onnx] Export finished. Verifying against PyTorch output...")

    verify(args.output, x_sam.cpu().numpy(), x_cnx.cpu().numpy(), torch_out)


def verify(onnx_path: str, x_sam_np: np.ndarray, x_cnx_np: np.ndarray, torch_out: np.ndarray) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export_onnx] onnxruntime not installed — skipping verification.")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"x_sam": x_sam_np, "x_cnx": x_cnx_np})[0]

    max_abs_diff = np.max(np.abs(onnx_out - torch_out))
    mean_abs_diff = np.mean(np.abs(onnx_out - torch_out))
    close = np.allclose(onnx_out, torch_out, atol=1e-3, rtol=1e-3)

    print(f"[verify] output shape      : {onnx_out.shape}")
    print(f"[verify] max abs diff      : {max_abs_diff:.6f}")
    print(f"[verify] mean abs diff     : {mean_abs_diff:.6f}")
    print(f"[verify] allclose(1e-3)    : {close}")

    if close:
        print("[verify] ✓ ONNX export matches PyTorch output")
    else:
        print("[verify] ✗ MISMATCH — do not deploy this ONNX file as-is")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export sam2unet_conv_bghr to ONNX")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained sam2unet_conv_bghr checkpoint (.pt)")
    p.add_argument("--sam2_ckpt", type=str, default=None,
                   help="Path to SAM2 Hiera-L checkpoint (only needed if the "
                        "trained checkpoint doesn't already include SAM2 weights)")
    p.add_argument("--output", type=str, default="sam2unet_conv_bghr.onnx")
    p.add_argument("--opset", type=int, default=19)
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
