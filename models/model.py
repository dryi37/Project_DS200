import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import trunc_normal_
from sam2.build_sam import build_sam2


# ---------------------------------------------------------------------------
# 1.  Adapter — lightweight prompt-learning wrapper for each SAM2 Hiera block
# ---------------------------------------------------------------------------

class Adapter(nn.Module):
    """
    Wraps a single SAM2 Hiera transformer block with a bottleneck prompt path.
    The prompt is added to the input residually before the main block forward.
    This keeps SAM2 weights frozen while allowing gradient flow through the
    small Linear layers.
    """

    def __init__(self, blk: nn.Module) -> None:
        super().__init__()
        self.block = blk
        dim: int = blk.attn.qkv.in_features          # infer from block

        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU(),
        )
        self._init_weights()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prompt = self.prompt_learn(x)
        return self.block(x + prompt)

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

        self.prompt_learn.apply(_init)


# ---------------------------------------------------------------------------
# 2.  UNet building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()
        mid = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Up(nn.Module):
    """Bilinear up-sampling followed by DoubleConv; optional skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x2 is not None:
            # Pad x2 to match x1 spatial size (handles odd dimensions)
            diffY = x1.size(2) - x2.size(2)
            diffX = x1.size(3) - x2.size(3)
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])
            x1 = torch.cat([x1, x2], dim=1)
        return self.conv(self.up(x1))


# ---------------------------------------------------------------------------
# 3.  Main model
# ---------------------------------------------------------------------------

class SAM2UNeXT(nn.Module):
    """
    SAM2-UNeXT with ConvNeXt-V2-Tiny as auxiliary encoder.

    Parameters
    ----------
    sam2_checkpoint : str | None
        Path to SAM2 Hiera-L checkpoint (.pt).  If None, SAM2 is loaded
        without pre-trained weights (useful for debugging).
    sam2_cfg : str
        SAM2 model config name (default: "sam2_hiera_l.yaml").
    convnext_pretrained : bool
        Whether to load ImageNet-22k→1k fine-tuned ConvNeXt-V2 weights.
    """

    # SAM2 Hiera-L produces four feature maps at these channel widths:
    _SAM2_CHANNELS = (144, 288, 576, 1152)   # stage0 … stage3
    # ConvNeXt-V2-Tiny final stage (stage 3) output channels:
    _CONVNEXT_FINAL_CH = 768

    def __init__(
        self,
        sam2_checkpoint: str | None = None,
        sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        convnext_pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # A) SAM2 — keep only the image encoder trunk; discard everything else
        # ------------------------------------------------------------------
        model = build_sam2(sam2_cfg, sam2_checkpoint)

        # Remove unused sub-modules to save VRAM
        for attr in (
            "sam_mask_decoder",
            "sam_prompt_encoder",
            "memory_encoder",
            "memory_attention",
            "mask_downsample",
            "obj_ptr_tpos_proj",
            "obj_ptr_proj",
        ):
            if hasattr(model, attr):
                delattr(model, attr)

        # Also drop the FPN neck — we want raw Hiera trunk features
        if hasattr(model.image_encoder, "neck"):
            del model.image_encoder.neck

        self.sam = model.image_encoder.trunk   # Hiera transformer trunk

        # Freeze *all* SAM2 parameters
        for param in self.sam.parameters():
            param.requires_grad = False

        # Wrap each transformer block with an Adapter for lightweight tuning
        self.sam.blocks = nn.Sequential(
            *[Adapter(blk) for blk in self.sam.blocks]
        )

        # ------------------------------------------------------------------
        # B) ConvNeXt-V2-Tiny auxiliary encoder
        #    features_only=True returns list: [stage0, stage1, stage2, stage3]
        #    channel widths  :                [ 96,    192,    384,    768  ]
        # ------------------------------------------------------------------
        self.convnext = timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=convnext_pretrained,
            features_only=True,
        )

        # Partial freeze: stem + stage 0 + stage 1 frozen; stage 2 + 3 unfrozen
        self._freeze_convnext_partial()

        # ------------------------------------------------------------------
        # C) Dense Glue Layer
        #    Projects ConvNeXt's final feature map (768 ch) to match each of
        #    SAM2's four skip-connection channel counts via 1×1 convolutions.
        # ------------------------------------------------------------------
        in_ch = self._CONVNEXT_FINAL_CH          # 768

        self.align1 = nn.Conv2d(in_ch, self._SAM2_CHANNELS[0], 1)   # → 144
        self.align2 = nn.Conv2d(in_ch, self._SAM2_CHANNELS[1], 1)   # → 288
        self.align3 = nn.Conv2d(in_ch, self._SAM2_CHANNELS[2], 1)   # → 576
        self.align4 = nn.Conv2d(in_ch, self._SAM2_CHANNELS[3], 1)   # → 1152

        # ------------------------------------------------------------------
        # D) Reduce blocks — merge SAM2 skip + aligned ConvNeXt feature
        # ------------------------------------------------------------------
        self.reduce1 = nn.Conv2d(self._SAM2_CHANNELS[0] * 2, 128, 1)
        self.reduce2 = nn.Conv2d(self._SAM2_CHANNELS[1] * 2, 128, 1)
        self.reduce3 = nn.Conv2d(self._SAM2_CHANNELS[2] * 2, 128, 1)
        self.reduce4 = nn.Conv2d(self._SAM2_CHANNELS[3] * 2, 128, 1)

        # ------------------------------------------------------------------
        # E) UNet-style decoder
        # ------------------------------------------------------------------
        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)

        # Final 1×1 conv → binary logit
        self.head = nn.Conv2d(128, 1, 1)

    # ------------------------------------------------------------------
    # Freezing helper
    # ------------------------------------------------------------------
    def _freeze_convnext_partial(self) -> None:
        """
        Freeze ConvNeXt-V2-Tiny stem + stages 0 & 1.
        Leave stages 2 & 3 trainable for domain adaptation.
        """
        # Duyệt qua tất cả các tham số của model bằng tên
        for name, param in self.convnext.named_parameters():
            # Nếu tên tham số chứa 'stem', 'stages.0' hoặc 'stages.1' thì đóng băng
            if name.startswith("stem") or name.startswith("stages.0") or name.startswith("stages.1"):
                param.requires_grad = False
            else:
                param.requires_grad = True # Mở khóa stages 2 & 3

        # Confirm trainable/frozen split
        total   = sum(p.numel() for p in self.convnext.parameters())
        frozen  = sum(p.numel() for p in self.convnext.parameters()
                      if not p.requires_grad)
        print(
            f"[ConvNeXt-V2] {frozen:,} / {total:,} params frozen "
            f"({100 * frozen / total:.1f} %)"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x_sam: torch.Tensor,
        x_cnx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_sam : (B, 3, 1024, 1024)  — high-res input for SAM2
        x_cnx : (B, 3,  448,  448)  — low-res input for ConvNeXt-V2

        Returns
        -------
        logits : (B, 1, 352, 352)   — raw (un-sigmoided) predictions
        """
        # --- SAM2 trunk → 4 hierarchical feature maps ---
        x1_s, x2_s, x3_s, x4_s = self.sam(x_sam)
        # shapes (B, 144, H1, W1), (B, 288, H2, W2), (B, 576, H3, W3), (B, 1152, H4, W4)

        # --- ConvNeXt-V2 → take only the final stage feature (index -1) ---
        x_cnx_feats = self.convnext(x_cnx)   # list of 4 feature maps
        x_c = x_cnx_feats[-1]                # (B, 768, h, w)  — stage 3 output

        # --- Dense Glue: project + spatially align to each SAM2 stage size ---
        x1_c = F.interpolate(self.align1(x_c), size=x1_s.shape[-2:], mode="bilinear", align_corners=False)
        x2_c = F.interpolate(self.align2(x_c), size=x2_s.shape[-2:], mode="bilinear", align_corners=False)
        x3_c = F.interpolate(self.align3(x_c), size=x3_s.shape[-2:], mode="bilinear", align_corners=False)
        x4_c = F.interpolate(self.align4(x_c), size=x4_s.shape[-2:], mode="bilinear", align_corners=False)

        # --- Fuse via concatenation + channel reduction ---
        x1 = self.reduce1(torch.cat([x1_s, x1_c], dim=1))
        x2 = self.reduce2(torch.cat([x2_s, x2_c], dim=1))
        x3 = self.reduce3(torch.cat([x3_s, x3_c], dim=1))
        x4 = self.reduce4(torch.cat([x4_s, x4_c], dim=1))

        # --- UNet decoder (coarse → fine) ---
        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        # --- Head + upsample to target output resolution (352×352) ---
        out = F.interpolate(
            self.head(x),
            size=(352, 352),
            mode="bilinear",
            align_corners=False,
        )
        return out   # (B, 1, 352, 352)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("SAM2-UNeXT + ConvNeXt-V2-Tiny — forward-pass smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Instantiate WITHOUT SAM2 checkpoint for a lightweight offline test.
    # Replace sam2_checkpoint with a real path in production.
    try:
        model = SAM2UNeXT(
            sam2_checkpoint=None,          # set to "/path/to/sam2_hiera_l.pt"
            convnext_pretrained=True,
        ).to(device)
    except Exception as exc:
        print(f"[ERROR] Model init failed: {exc}")
        sys.exit(1)

    model.eval()

    B = 1
    x_sam = torch.randn(B, 3, 1024, 1024, device=device)
    x_cnx = torch.randn(B, 3,  448,  448, device=device)

    with torch.no_grad():
        out = model(x_sam, x_cnx)

    print(f"Input  SAM2       : {tuple(x_sam.shape)}")
    print(f"Input  ConvNeXt   : {tuple(x_cnx.shape)}")
    print(f"Output logits     : {tuple(out.shape)}")
    assert out.shape == (B, 1, 352, 352), "Unexpected output shape!"
    print("✓  Forward pass OK — output shape (B, 1, 352, 352) verified.")

    # Parameter count summary
    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params   : {total_params - trainable_params:,}")
