import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2


class DoubleConv(nn.Module):
    """Conv-BN-ReLU x 2, same style as original SAM2-UNet decoder."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upsample, concatenate skip feature, then DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    """Same lightweight adapter idea as original SAM2-UNet."""
    def __init__(self, blk) -> None:
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features

        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU(),
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        x = x + prompt
        return self.block(x)


class BasicConv2d(nn.Module):
    """
    Original SAM2-UNet/PraNet-style BasicConv2d for RFB:
    Conv + BN, no ReLU here.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBNReLU(nn.Module):
    """Used only in the new boundary/refinement branch."""
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RFB_modified(nn.Module):
    """Same RFB structure as original SAM2-UNet."""
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )

        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class BoundaryGuidedRefinement(nn.Module):
    """
    New lightweight branch:
    1) predict edge from high-resolution decoder feature
    2) turn edge probability into attention
    3) refine final mask feature
    """
    def __init__(self, channels=64):
        super().__init__()

        self.edge_head = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

        self.edge_attention = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
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

        refined_feat = self.refine(torch.cat([feat, edge_prob], dim=1))
        return refined_feat, edge_logit


class SAM2UNet_BGHR(nn.Module):
    """
    Boundary-Guided High-Resolution SAM2-UNet.

    Outputs:
        out_final  : final refined mask logit, input resolution
        out_side1  : deep supervision side output from decoder stage 1
        out_side2  : deep supervision side output from decoder stage 2
        out_edge   : predicted boundary logit, input resolution
        out_coarse : coarse mask before boundary refinement, input resolution
    """
    def __init__(self, checkpoint_path=None) -> None:
        super().__init__()

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)

        # Remove SAM2 parts not used in U-Net segmentation.
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck

        self.encoder = model.image_encoder.trunk

        # Freeze pretrained SAM2-Hiera encoder.
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Insert trainable adapters into Hiera blocks.
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(Adapter(block))
        self.encoder.blocks = nn.Sequential(*blocks)

        # SAM2-Hiera-L channels.
        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)

        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)

        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)

        self.coarse_head = nn.Conv2d(64, 1, kernel_size=1)
        self.boundary_refine = BoundaryGuidedRefinement(channels=64)
        self.final_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        x1, x2, x3, x4 = self.encoder(x)

        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)
        x4 = self.rfb4(x4)

        d1 = self.up1(x4, x3)
        out_side1 = self.side1(d1)

        d2 = self.up2(d1, x2)
        out_side2 = self.side2(d2)

        d3 = self.up3(d2, x1)

        out_coarse = self.coarse_head(d3)
        refined_feat, out_edge = self.boundary_refine(d3)
        out_final = self.final_head(refined_feat)

        # Make output size robust for 352, 448, 512, or multiscale.
        out_final = F.interpolate(out_final, size=input_size, mode="bilinear", align_corners=False)
        out_side1 = F.interpolate(out_side1, size=input_size, mode="bilinear", align_corners=False)
        out_side2 = F.interpolate(out_side2, size=input_size, mode="bilinear", align_corners=False)
        out_edge = F.interpolate(out_edge, size=input_size, mode="bilinear", align_corners=False)
        out_coarse = F.interpolate(out_coarse, size=input_size, mode="bilinear", align_corners=False)

        return out_final, out_side1, out_side2, out_edge, out_coarse


if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet_BGHR().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        outs = model(x)
        print([o.shape for o in outs])
