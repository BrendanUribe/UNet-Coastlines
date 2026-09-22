"""U-Net with a segmentation head and a deeply-supervised boundary head.

Architecture notes and why each choice was made
-----------------------------------------------
The skeleton is still the original U-Net (Ronneberger, Fischer & Brox, MICCAI
2015, arXiv:1505.04597).  Four changes to the 2015 formulation:

* **GroupNorm** (Wu & He, ECCV 2018, arXiv:1803.08494) instead of no
  normalisation.  The original had none because BatchNorm was not yet standard,
  and the consequence here was a learning rate pinned at 5e-5 to stay stable.
  GroupNorm rather than BatchNorm because batch sizes for 240x320 imagery are
  small (4-8), which is exactly where BatchNorm's batch statistics degrade.

* **Depth 4 by default** instead of 3.  A whole-disc Earth view needs global
  context - which side is lit, where the limb runs - and three poolings leaves
  the bottleneck receptive field too small to see it.  nnU-Net (Isensee et al.,
  Nature Methods 18, 203-211, 2021) derives depth from patch size and is the
  reference for configuring a plain U-Net properly rather than reaching for an
  exotic variant.

* **Bilinear upsample + 3x3 conv** instead of ConvTranspose2d.  Transposed
  convolutions produce periodic checkerboard artifacts (Odena, Dumoulin & Olah,
  Distill 2016).  With kernel=2/stride=2 the original was at the least-bad
  setting, but the output here is one-pixel-wide lines, which is precisely the
  case where periodic artifacts are unaffordable.

* **Deep supervision on the boundary head.**  Edge detection benefits strongly
  from supervising every scale and fusing, which is the central result of HED
  (Xie & Tu, ICCV 2015, arXiv:1504.06375).  The combined segmentation + edge
  U-Net for coastlines is HED-UNet (Heidler et al., IEEE TGRS 60, 2022,
  arXiv:2103.01849), which is the closest published analogue to this model.

The boundary head emits three channels - coastline, limb and terminator - see
``labels.py`` for why those are separated rather than merged into one
"coastline" output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int, groups: int = 8) -> nn.Module:
    """GroupNorm with a group count that always divides the channel count."""
    g = min(groups, channels)
    while channels % g:
        g -= 1
    return nn.GroupNorm(g, channels)


class DoubleConv(nn.Module):
    """(conv 3x3 -> norm -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels, groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels, groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Bilinear upsample, concatenate the skip, then DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, groups)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """Two-headed U-Net.

    Parameters
    ----------
    in_channels
        Input image channels (3 for RGB).
    num_classes
        Segmentation classes.  Default 5: space/water/land/cloud/night.
    num_edge_channels
        Boundary channels.  Default 3: coastline/limb/terminator.
    base_width
        Channels at the finest level.  Halve it (32) for an embedded target;
        see the deployment notes in the README.
    depth
        Number of downsampling steps.
    deep_supervision
        Emit per-scale boundary logits during training, as in HED.  They are
        returned only in training mode and are ignored at inference.

    Returns from ``forward``
    ------------------------
    ``seg_out``  ``(B, num_classes, H, W)`` logits
    ``edge_out`` ``(B, num_edge_channels, H, W)`` logits
    ``aux``      list of ``(B, num_edge_channels, H, W)`` logits, one per decoder
                 scale, upsampled to full resolution.  Empty unless training
                 with ``deep_supervision``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        num_edge_channels: int = 3,
        base_width: int = 64,
        depth: int = 4,
        groups: int = 8,
        deep_supervision: bool = True,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.num_edge_channels = num_edge_channels

        widths = [base_width * 2 ** i for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for w in widths:
            self.encoders.append(DoubleConv(prev, w, groups))
            prev = w

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(prev, prev * 2, groups)

        self.decoders = nn.ModuleList()
        prev = prev * 2
        for w in reversed(widths):
            self.decoders.append(Up(prev, w, w, groups))
            prev = w

        self.seg_head = nn.Conv2d(widths[0], num_classes, kernel_size=1)
        self.edge_head = nn.Conv2d(widths[0], num_edge_channels, kernel_size=1)

        # HED-style side outputs, one per decoder scale except the finest
        # (which the main edge head already covers).
        self.side_heads = nn.ModuleList(
            nn.Conv2d(w, num_edge_channels, kernel_size=1)
            for w in list(reversed(widths))[:-1]
        ) if deep_supervision else nn.ModuleList()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def size_divisor(self) -> int:
        """Input H and W must be multiples of this."""
        return 2 ** self.depth

    def forward(self, x):
        h, w = x.shape[-2:]
        d = self.size_divisor
        if h % d or w % d:
            raise ValueError(
                f"input is {h}x{w}, but a depth-{self.depth} U-Net needs both "
                f"dimensions divisible by {d}. Resize or pad first "
                f"(240x320 is the project default and satisfies this)."
            )

        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        aux = []
        for i, (dec, skip) in enumerate(zip(self.decoders, reversed(skips))):
            x = dec(x, skip)
            if self.deep_supervision and self.training and i < len(self.side_heads):
                side = self.side_heads[i](x)
                aux.append(F.interpolate(side, size=(h, w), mode="bilinear",
                                         align_corners=False))

        return self.seg_head(x), self.edge_head(x), aux
