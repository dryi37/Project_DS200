import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import trunc_normal_
from sam2.build_sam import build_sam2


class Adapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32), nn.GELU(),
            nn.Linear(32, dim), nn.GELU(),
        )
        self._init_weights()

    def forward(self, x):
        return self.block(x + self.prompt_learn(x))

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
        self.prompt_learn.apply(_init)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size(2) - x2.size(2)
            diffX = x1.size(3) - x2.size(3)
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])
            x1 = torch.cat([x1, x2], dim=1)
        return self.conv(self.up(x1))


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BoundaryGuidedRefinement(nn.Module):
    """Predicts edge map → edge attention → refines feature for final mask."""
    def __init__(self, channels=128):
        super().__init__()
        self.edge_head = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.edge_attention = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ConvBNReLU(channels + 1, channels, kernel_size=3, padding=1),
            ConvBNReLU(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, feat):
        edge_logit = self.edge_head(feat)
        edge_prob = torch.sigmoid(edge_logit)
        att = self.edge_attention(edge_prob)
        feat = feat + feat * att
        refined = self.refine(torch.cat([feat, edge_prob], dim=1))
        return refined, edge_logit


class SAM2UNeXT_BG(nn.Module):
    """
    SAM2-UNeXT + BGHR. Encoder/fusion/decoder identical to original SAM2UNeXT;
    BGHR adds deep supervision + edge prediction + refinement.

    Forward signature: model(x_sam, x_cnx)  -> dict.
    Returns dict with: final, coarse, s1, s2, edge — all (B, 1, 352, 352).
    """
    _SAM2_CHANNELS = (144, 288, 576, 1152)
    _CONVNEXT_FINAL_CH = 768

    def __init__(self,
                 sam2_checkpoint=None,
                 sam2_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
                 convnext_pretrained=True):
        super().__init__()

        # ---------- SAM2 Hiera-L (frozen + adapters) ----------
        model = build_sam2(sam2_cfg, sam2_checkpoint)
        for attr in ("sam_mask_decoder", "sam_prompt_encoder", "memory_encoder",
                     "memory_attention", "mask_downsample", "obj_ptr_tpos_proj",
                     "obj_ptr_proj"):
            if hasattr(model, attr):
                delattr(model, attr)
        if hasattr(model.image_encoder, "neck"):
            del model.image_encoder.neck

        self.sam = model.image_encoder.trunk
        for p in self.sam.parameters():
            p.requires_grad = False
        self.sam.blocks = nn.Sequential(*[Adapter(blk) for blk in self.sam.blocks])

        # ---------- ConvNeXt-V2-Tiny (partial freeze: stage 2/3 unfrozen) ----------
        self.convnext = timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=convnext_pretrained, features_only=True,
        )
        self._freeze_convnext_partial()

        # ---------- Dense Glue (project ConvNeXt 768 → SAM2 channel widths) ----------
        in_ch = self._CONVNEXT_FINAL_CH
        self.align1 = nn.Conv2d(in_ch, self._SAM2_CHANNELS[0], 1)
        self.align2 = nn.Conv2d(in_ch, self._SAM2_CHANNELS[1], 1)
        self.align3 = nn.Conv2d(in_ch, self._SAM2_CHANNELS[2], 1)
        self.align4 = nn.Conv2d(in_ch, self._SAM2_CHANNELS[3], 1)

        # ---------- Reduce to 128 ch ----------
        self.reduce1 = nn.Conv2d(self._SAM2_CHANNELS[0] * 2, 128, 1)
        self.reduce2 = nn.Conv2d(self._SAM2_CHANNELS[1] * 2, 128, 1)
        self.reduce3 = nn.Conv2d(self._SAM2_CHANNELS[2] * 2, 128, 1)
        self.reduce4 = nn.Conv2d(self._SAM2_CHANNELS[3] * 2, 128, 1)

        # ---------- UNet decoder ----------
        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)

        # ---------- BGHR heads (the new contributions) ----------
        self.side2       = nn.Conv2d(128, 1, kernel_size=1)
        self.side1       = nn.Conv2d(128, 1, kernel_size=1)
        self.coarse_head = nn.Conv2d(128, 1, kernel_size=1)
        self.boundary_refine = BoundaryGuidedRefinement(channels=128)
        self.final_head  = nn.Conv2d(128, 1, kernel_size=1)

    def _freeze_convnext_partial(self):
        for name, p in self.convnext.named_parameters():
            if name.startswith("stem") or name.startswith("stages.0") or name.startswith("stages.1"):
                p.requires_grad = False
            else:
                p.requires_grad = True
        total  = sum(p.numel() for p in self.convnext.parameters())
        frozen = sum(p.numel() for p in self.convnext.parameters() if not p.requires_grad)
        print(f"[ConvNeXt-V2] {frozen:,} / {total:,} params frozen "
              f"({100*frozen/total:.1f}%)")

    def forward(self, x_sam, x_cnx):
        # Encoders
        x1_s, x2_s, x3_s, x4_s = self.sam(x_sam)
        x_c = self.convnext(x_cnx)[-1]

        # Dense glue
        x1_c = F.interpolate(self.align1(x_c), size=x1_s.shape[-2:], mode="bilinear", align_corners=False)
        x2_c = F.interpolate(self.align2(x_c), size=x2_s.shape[-2:], mode="bilinear", align_corners=False)
        x3_c = F.interpolate(self.align3(x_c), size=x3_s.shape[-2:], mode="bilinear", align_corners=False)
        x4_c = F.interpolate(self.align4(x_c), size=x4_s.shape[-2:], mode="bilinear", align_corners=False)

        # Fuse
        x1 = self.reduce1(torch.cat([x1_s, x1_c], dim=1))
        x2 = self.reduce2(torch.cat([x2_s, x2_c], dim=1))
        x3 = self.reduce3(torch.cat([x3_s, x3_c], dim=1))
        x4 = self.reduce4(torch.cat([x4_s, x4_c], dim=1))

        # Decoder + deep supervision
        d = self.up4(x4)
        d = self.up3(d, x3)
        s2 = self.side2(d)
        d = self.up2(d, x2)
        s1 = self.side1(d)
        d = self.up1(d, x1)

        # BGHR
        coarse = self.coarse_head(d)
        refined, edge = self.boundary_refine(d)
        final = self.final_head(refined)

        # Upsample all to 352x352
        T = (352, 352)
        return {
            "final":  F.interpolate(final,  size=T, mode="bilinear", align_corners=False),
            "coarse": F.interpolate(coarse, size=T, mode="bilinear", align_corners=False),
            "s1":     F.interpolate(s1,     size=T, mode="bilinear", align_corners=False),
            "s2":     F.interpolate(s2,     size=T, mode="bilinear", align_corners=False),
            "edge":   F.interpolate(edge,   size=T, mode="bilinear", align_corners=False),
        }


def load_friend_checkpoint(model, ckpt_path, verbose=True):
    """
    Load friend's SAM2UNeXT checkpoint into SAM2UNeXT_BG.
    Matches keys with same name AND same shape; skips BGHR-specific keys
    (side1, side2, coarse_head, boundary_refine, final_head) plus friend's
    'head' layer (which we don't use).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd_src = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    sd_dst = model.state_dict()
    loaded, mismatched, missing = 0, 0, 0
    for k, v in sd_src.items():
        if k in sd_dst and sd_dst[k].shape == v.shape:
            sd_dst[k] = v
            loaded += 1
        elif k in sd_dst:
            mismatched += 1
        else:
            missing += 1
    model.load_state_dict(sd_dst)
    if verbose:
        new_keys = sum(1 for k in sd_dst if k not in sd_src)
        print(f"[load_friend_checkpoint] loaded={loaded}  "
              f"src_extra={missing} (e.g. friend's 'head')  "
              f"dst_new={new_keys} (BGHR heads, random init)")
    return model


if __name__ == "__main__":
    import sys
    print("SAM2-UNeXT-BG smoke test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = SAM2UNeXT_BG(sam2_checkpoint=None, convnext_pretrained=True).to(device)
    except Exception as e:
        print(f"[ERROR] {e}"); sys.exit(1)
    model.eval()
    x_sam = torch.randn(1, 3, 1024, 1024, device=device)
    x_cnx = torch.randn(1, 3,  448,  448, device=device)
    with torch.no_grad():
        out = model(x_sam, x_cnx)
    for k, v in out.items():
        print(f"  {k:10s}: {tuple(v.shape)}")
    total = sum(p.numel() for p in model.parameters()) / 1e6
    train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total: {total:.2f}M | Trainable: {train:.2f}M")
