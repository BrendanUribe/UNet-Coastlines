# HED-UNet model - predicts land/water mask AND coastline edge at the same time
# adapted from https://github.com/khdlr/HED-UNet (MIT License, Copyright (c) 2021 Konrad Heidler)
# paper: Heidler et al. 2021, "HED-UNet: Combined Segmentation and Edge Detection for Monitoring the Antarctic Coastline"

import torch # py torch deep learning library
import torch.nn as nn # neural network tools
import torch.nn.functional as F # resizing (interpolate), softmax


# double convolution block (same idea as DoubleConv in UNet.py but with optional batch norm)
class Convx2(nn.Module):
    def __init__(self, c_in, c_out, bn, padding_mode='zeros'):
        super().__init__()
        conv_args = dict(padding=1, padding_mode=padding_mode, bias=not bn) # 3x3 kernel, padding 1 to not shrink
        self.conv1 = nn.Conv2d(c_in, c_out, 3, **conv_args)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, **conv_args)
        self.bn1 = nn.BatchNorm2d(c_out) if bn else nn.Identity() # batch norm keeps training stable
        self.bn2 = nn.BatchNorm2d(c_out) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# encoder block - halves image size with a strided conv (instead of max pool) then double conv
class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, bn=True, padding_mode='zeros'):
        super().__init__()
        self.convdown = nn.Conv2d(c_in, c_in, 2, stride=2, bias=not bn)
        self.bn = nn.BatchNorm2d(c_in) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv_block = Convx2(c_in, c_out, bn=bn, padding_mode=padding_mode)

    def forward(self, x):
        x = self.relu(self.bn(self.convdown(x)))
        return self.conv_block(x)


# decoder block - doubles image size, adds skip connection, then double conv
class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, bn=True, padding_mode='zeros'):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, 2, stride=2, bias=not bn)
        self.bn = nn.BatchNorm2d(c_in // 2) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv_block = Convx2(c_in, c_out, bn=bn, padding_mode=padding_mode)

    def forward(self, x, skip):
        x = self.relu(self.bn(self.up(x)))
        x = torch.cat([x, skip], dim=1) # skip connection like UNet.py
        return self.conv_block(x)


class HEDUNet(nn.Module): # main HED-UNet model
    # in_channels 3 for rgb
    # out_channels 2: channel 0 = land/water mask, channel 1 = coastline edge
    # stack_height = number of down/up levels, image size must be divisible by 2**stack_height (256 / 32 = 8 ok)
    def __init__(self, in_channels=3, out_channels=2, base_channels=16, stack_height=5, batch_norm=True):
        super().__init__()
        bc = base_channels
        self.out_channels = out_channels
        conv_args = dict(bn=batch_norm, padding_mode='replicate')

        self.init = nn.Conv2d(in_channels, bc, 1) # rgb to base feature maps

        # Encoder - 16 -> 32 -> 64 -> 128 -> 256 -> 512 feature maps
        self.down_blocks = nn.ModuleList([
            DownBlock((1 << i) * bc, (2 << i) * bc, **conv_args)
            for i in range(stack_height)
        ])

        # Decoder - 512 -> 256 -> ... -> 16 feature maps
        self.up_blocks = nn.ModuleList([
            UpBlock((2 << i) * bc, (1 << i) * bc, **conv_args)
            for i in reversed(range(stack_height))
        ])

        # one small prediction head per decoder level (bottleneck + each up block)
        # each gives a mask + edge guess at its own scale (deep supervision)
        self.predictors = nn.ModuleList([
            nn.Conv2d((1 << i) * bc, out_channels, 1)
            for i in reversed(range(stack_height + 1))
        ])

        # attention - learns how much to trust each scale per pixel when merging them
        self.queries = nn.ModuleList([
            nn.Conv2d((1 << i) * bc, out_channels, 1)
            for i in reversed(range(stack_height + 1))
        ])

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.init(x)

        # Encoder
        skip_connections = []
        for block in self.down_blocks:
            skip_connections.append(x)
            x = block(x)

        # Decoder - keep features from every level
        multilevel_features = [x]
        for block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = block(x, skip)
            multilevel_features.append(x)

        # predictions at every level (coarse -> fine), plus upsampled to full size
        predictions_list = []
        full_scale_preds = []
        for feature_map, predictor in zip(multilevel_features, self.predictors):
            prediction = predictor(feature_map)
            predictions_list.append(prediction)
            full_scale_preds.append(F.interpolate(prediction, size=(H, W), mode='bilinear', align_corners=True))
        predictions = torch.cat(full_scale_preds, dim=1)

        # merge all levels with attention weights (softmax across levels)
        queries = [F.interpolate(q(feat), size=(H, W), mode='bilinear', align_corners=True)
                   for q, feat in zip(self.queries, multilevel_features)]
        queries = torch.cat(queries, dim=1).reshape(B, -1, self.out_channels, H, W)
        attn = F.softmax(queries, dim=1)
        predictions = predictions.reshape(B, -1, self.out_channels, H, W)
        combined_prediction = torch.sum(attn * predictions, dim=1)

        # combined = final output logits [B, 2, H, W]
        # levels = list of per-level logits, finest (full size) first, used for deep supervision loss
        return combined_prediction, list(reversed(predictions_list))
