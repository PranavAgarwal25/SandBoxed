import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
#  Floorplan element classes – shared between the cloud detection service
#  and the local CNN so that both backends produce outputs in the same
#  label space.
# ---------------------------------------------------------------------------
FLOORPLAN_CLASSES = [
    "background",       # 0
    "space_balconi",     # 1
    "space_bedroom",     # 2
    "space_corridor",    # 3
    "space_dining",      # 4
    "space_kitchen",     # 5
    "space_laundry",     # 6
    "space_living",      # 7
    "space_lobby",       # 8
    "space_office",      # 9
    "space_other",       # 10
    "space_parking",     # 11
    "space_staircase",   # 12
    "space_toilet",      # 13
]

NUM_CLASSES = len(FLOORPLAN_CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(FLOORPLAN_CLASSES)}

# Default input resolution expected by every network variant
INPUT_SIZE = (512, 512)


# ===================================================================== #
#                        BUILDING  BLOCKS                                #
# ===================================================================== #

class ConvBlock(nn.Module):
    """Two successive 3x3 convolutions each followed by BatchNorm and ReLU.

    This is the fundamental repeating unit used in every encoder and decoder
    stage.  Bias is disabled in the convolution because BatchNorm already
    absorbs the shift.
    """

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention.

    Compresses spatial dimensions via global average pooling, passes
    through a two-layer bottleneck, and produces per-channel scaling
    factors.  Applied after the ConvBlock in attention-enabled encoder
    stages.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """Lightweight spatial attention gate.

    Concatenates channel-wise max and mean pooling, feeds through a 7x7
    convolution, and produces a spatial weighting mask.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True)[0]
        combined = torch.cat([avg_pool, max_pool], dim=1)
        return x * self.conv(combined)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ResidualConvBlock(nn.Module):
    """ConvBlock with an additive residual (identity / 1x1 projection)."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


# ===================================================================== #
#                     ENCODER / DECODER  STAGES                          #
# ===================================================================== #

class EncoderBlock(nn.Module):
    """Encoder stage: ConvBlock -> optional CBAM -> 2x2 max-pool.

    Returns both the feature map (for skip connections) and the
    down-sampled output.
    """

    def __init__(self, in_ch, out_ch, use_attention=False, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)
        self.attention = CBAM(out_ch) if use_attention else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        features = self.attention(features)
        pooled = self.pool(features)
        return features, pooled


class ResidualEncoderBlock(nn.Module):
    """Encoder stage with a residual connection inside the conv block."""

    def __init__(self, in_ch, out_ch, use_attention=False, dropout=0.0):
        super().__init__()
        self.conv = ResidualConvBlock(in_ch, out_ch, dropout=dropout)
        self.attention = CBAM(out_ch) if use_attention else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        features = self.attention(features)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder stage: upsample -> concatenate skip -> ConvBlock.

    Handles spatial-dimension mismatches that arise when the input
    resolution is not a power of two.
    """

    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionGateDecoderBlock(nn.Module):
    """Decoder with a soft attention gate on the skip connection.

    The gate learns to suppress irrelevant regions in the encoder feature
    map before concatenation, which helps the network focus on wall
    boundaries and room interiors.
    """

    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        inter_ch = max((in_ch // 2 + skip_ch) // 4, 16)
        self.gate_w_g = nn.Conv2d(in_ch // 2, inter_ch, kernel_size=1, bias=False)
        self.gate_w_x = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.gate_psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.gate_relu = nn.ReLU(inplace=True)

        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])

        # Attention gate
        g = self.gate_w_g(x)
        s = self.gate_w_x(skip)
        psi = self.gate_psi(self.gate_relu(g + s))
        skip = skip * psi

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ===================================================================== #
#                       AUXILIARY  MODULES                                #
# ===================================================================== #

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context aggregation.

    Used as an alternative bottleneck that captures features at multiple
    receptive-field scales without increasing spatial resolution.
    """

    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=rate,
                              dilation=rate, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        )
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )

    def forward(self, x):
        outputs = []
        for conv in self.convs:
            out = conv(x)
            if out.size(2) != x.size(2) or out.size(3) != x.size(3):
                out = F.interpolate(out, size=x.shape[2:], mode="bilinear",
                                    align_corners=False)
            outputs.append(out)
        return self.project(torch.cat(outputs, dim=1))


class FeaturePyramidFusion(nn.Module):
    """Lightweight top-down feature pyramid that merges multi-scale encoder
    features into a single representation before the segmentation head.

    Expects a list of feature maps ordered from coarsest to finest.
    """

    def __init__(self, channels_list):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()
        for ch in channels_list:
            self.lateral_convs.append(
                nn.Conv2d(ch, channels_list[-1], kernel_size=1, bias=False)
            )
            self.smooth_convs.append(
                nn.Conv2d(channels_list[-1], channels_list[-1], kernel_size=3,
                          padding=1, bias=False)
            )

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:],
                               mode="bilinear", align_corners=False)
            laterals[i] = laterals[i] + up
        smoothed = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]
        target_size = smoothed[0].shape[2:]
        aligned = [
            F.interpolate(s, size=target_size, mode="bilinear", align_corners=False)
            if s.shape[2:] != target_size else s
            for s in smoothed
        ]
        return sum(aligned)


# ===================================================================== #
#                       FULL  NETWORK  VARIANTS                          #
# ===================================================================== #

class FloorplanSegmentationNet(nn.Module):
    """U-Net encoder-decoder for semantic segmentation of floorplan images.

    Produces a per-pixel class map over ``NUM_CLASSES`` floorplan element
    types (background, rooms, kitchen, toilet, …) which is later consumed
    by the FloorplanToBlenderLib pipeline to generate 3D geometry.

    Architecture
    ~~~~~~~~~~~~
    * Encoder : 4 down-sampling stages  (3->64->128->256->512)
    * Bottleneck : 512 -> 1024
    * Decoder : 4 up-sampling stages with skip connections
    * Head : 1x1 conv projecting to ``num_classes``
    """

    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, dropout=0.1,
                 use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64, use_attention=use_attention)
        self.enc2 = EncoderBlock(64, 128, use_attention=use_attention)
        self.enc3 = EncoderBlock(128, 256, use_attention=use_attention, dropout=dropout)
        self.enc4 = EncoderBlock(256, 512, use_attention=use_attention, dropout=dropout)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024, dropout=dropout)

        # Decoder
        self.dec4 = DecoderBlock(1024, 512, 512, dropout=dropout)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        # Segmentation head
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.seg_head(x)


class DepthEstimationNet(nn.Module):
    """Encoder-decoder that predicts a single-channel relative depth map.

    The depth values are used downstream to assign wall heights and floor
    elevation offsets when generating 3D geometry.  Outputs are in [0, 1]
    via a sigmoid activation.
    """

    def __init__(self, in_channels=3, dropout=0.1, use_attention=False):
        super().__init__()

        self.enc1 = EncoderBlock(in_channels, 64, use_attention=use_attention)
        self.enc2 = EncoderBlock(64, 128, use_attention=use_attention)
        self.enc3 = EncoderBlock(128, 256, use_attention=use_attention, dropout=dropout)
        self.enc4 = EncoderBlock(256, 512, use_attention=use_attention, dropout=dropout)

        self.bottleneck = ConvBlock(512, 1024, dropout=dropout)

        self.dec4 = DecoderBlock(1024, 512, 512, dropout=dropout)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        self.depth_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.depth_head(x)


class MultiTaskFloorplanNet(nn.Module):
    """Joint segmentation + depth estimation with a shared encoder.

    This is the primary model used in the pipeline.  A single forward pass
    produces both a semantic segmentation map (room types) and a relative
    depth map that guides 3D extrusion heights.

    Shared encoder
    ~~~~~~~~~~~~~~
    4-stage down-sampling identical to the standalone nets.

    Task-specific decoders
    ~~~~~~~~~~~~~~~~~~~~~~
    * **Segmentation decoder** – produces ``num_classes`` channels
      (trained with cross-entropy + Dice loss).
    * **Depth decoder** – produces 1 channel in [0, 1]
      (trained with L1 + gradient loss).

    Both decoders receive the same skip connections from the shared
    encoder, which encourages complementary feature learning.
    """

    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, dropout=0.15,
                 use_attention=True):
        super().__init__()
        self.num_classes = num_classes

        # Shared encoder
        self.enc1 = EncoderBlock(in_channels, 64, use_attention=use_attention)
        self.enc2 = EncoderBlock(64, 128, use_attention=use_attention)
        self.enc3 = EncoderBlock(128, 256, use_attention=use_attention, dropout=dropout)
        self.enc4 = EncoderBlock(256, 512, use_attention=use_attention, dropout=dropout)
        self.bottleneck = ConvBlock(512, 1024, dropout=dropout)

        # Segmentation decoder
        self.seg_dec4 = AttentionGateDecoderBlock(1024, 512, 512, dropout=dropout)
        self.seg_dec3 = AttentionGateDecoderBlock(512, 256, 256)
        self.seg_dec2 = DecoderBlock(256, 128, 128)
        self.seg_dec1 = DecoderBlock(128, 64, 64)
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

        # Depth decoder
        self.dep_dec4 = DecoderBlock(1024, 512, 512, dropout=dropout)
        self.dep_dec3 = DecoderBlock(512, 256, 256)
        self.dep_dec2 = DecoderBlock(256, 128, 128)
        self.dep_dec1 = DecoderBlock(128, 64, 64)
        self.dep_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Shared encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        b = self.bottleneck(x)

        # Segmentation branch
        seg = self.seg_dec4(b, s4)
        seg = self.seg_dec3(seg, s3)
        seg = self.seg_dec2(seg, s2)
        seg = self.seg_dec1(seg, s1)
        seg_out = self.seg_head(seg)

        # Depth branch
        dep = self.dep_dec4(b, s4)
        dep = self.dep_dec3(dep, s3)
        dep = self.dep_dec2(dep, s2)
        dep = self.dep_dec1(dep, s1)
        dep_out = self.dep_head(dep)

        return seg_out, dep_out


class FloorplanSegmentationNetV2(nn.Module):
    """Deeper segmentation network with residual encoder blocks and ASPP
    bottleneck for improved multi-scale reasoning on large floorplans.
    """

    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, dropout=0.15):
        super().__init__()

        self.enc1 = ResidualEncoderBlock(in_channels, 64, use_attention=True)
        self.enc2 = ResidualEncoderBlock(64, 128, use_attention=True)
        self.enc3 = ResidualEncoderBlock(128, 256, use_attention=True, dropout=dropout)
        self.enc4 = ResidualEncoderBlock(256, 512, use_attention=True, dropout=dropout)

        self.bottleneck = ASPP(512, 1024, rates=(6, 12, 18))

        self.dec4 = AttentionGateDecoderBlock(1024, 512, 512, dropout=dropout)
        self.dec3 = AttentionGateDecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)

        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.seg_head(x)


# ===================================================================== #
#                          LOSS  FUNCTIONS                                #
# ===================================================================== #

class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation.

    Works well when classes are highly imbalanced (e.g. small toilet
    regions vs. large background).
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        num_classes = pred.size(1)
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (pred_soft * target_onehot).sum(dim=dims)
        cardinality = pred_soft.sum(dim=dims) + target_onehot.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss that down-weights well-classified pixels.

    Particularly useful for floorplan segmentation where background
    pixels dominate and small rooms may be under-represented.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


class SegmentationLoss(nn.Module):
    """Combined cross-entropy + Dice for class-imbalanced segmentation."""

    def __init__(self, class_weights=None, dice_weight=0.5, ce_weight=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, device=pred.device,
                                  dtype=pred.dtype)
        else:
            weight = None
        ce = F.cross_entropy(pred, target, weight=weight)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


class DepthGradientLoss(nn.Module):
    """L1 + image-gradient loss for depth prediction.

    The gradient term encourages sharp depth transitions at wall edges
    which is critical for correct 3D extrusion.
    """

    def __init__(self, gradient_weight=0.5):
        super().__init__()
        self.gradient_weight = gradient_weight

    @staticmethod
    def _gradient(tensor):
        dx = torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:])
        dy = torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :])
        return dx, dy

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)

        pred_dx, pred_dy = self._gradient(pred)
        tgt_dx, tgt_dy = self._gradient(target)
        grad_loss = F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)

        return l1 + self.gradient_weight * grad_loss


class BerHuLoss(nn.Module):
    """Reverse Huber (BerHu) loss for monocular depth estimation.

    L1 for small residuals, L2 for large residuals.  The threshold is
    determined adaptively per batch as 20 percent of the maximum residual.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        c = 0.2 * diff.max().detach()
        mask_l1 = diff <= c
        mask_l2 = ~mask_l1
        loss = diff[mask_l1].sum()
        if mask_l2.any():
            diff_l2 = diff[mask_l2]
            loss = loss + ((diff_l2 ** 2 + c ** 2) / (2.0 * c)).sum()
        return loss / diff.numel()


class MultiTaskLoss(nn.Module):
    """Combined loss for the multi-task network::

        total = seg_weight * (CE + Dice)(seg_pred, seg_target)
              + dep_weight * DepthGradient(dep_pred, dep_target)
    """

    def __init__(self, seg_weight=1.0, dep_weight=1.0, class_weights=None):
        super().__init__()
        self.seg_loss = SegmentationLoss(class_weights)
        self.dep_loss = DepthGradientLoss()
        self.seg_weight = seg_weight
        self.dep_weight = dep_weight

    def forward(self, seg_pred, seg_target, dep_pred, dep_target):
        ls = self.seg_loss(seg_pred, seg_target)
        ld = self.dep_loss(dep_pred, dep_target)
        return self.seg_weight * ls + self.dep_weight * ld, ls, ld


class UncertaintyMultiTaskLoss(nn.Module):
    """Learnable uncertainty-weighted multi-task loss (Kendall et al.).

    Instead of hand-tuning ``seg_weight`` and ``dep_weight``, the loss
    learns a log-variance parameter per task that automatically balances
    the two losses during training.
    """

    def __init__(self, class_weights=None):
        super().__init__()
        self.seg_loss = SegmentationLoss(class_weights)
        self.dep_loss = DepthGradientLoss()
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_dep = nn.Parameter(torch.zeros(1))

    def forward(self, seg_pred, seg_target, dep_pred, dep_target):
        ls = self.seg_loss(seg_pred, seg_target)
        ld = self.dep_loss(dep_pred, dep_target)

        precision_seg = torch.exp(-self.log_var_seg)
        precision_dep = torch.exp(-self.log_var_dep)

        total = (precision_seg * ls + self.log_var_seg +
                 precision_dep * ld + self.log_var_dep)
        return total, ls, ld


# ===================================================================== #
#                       METRIC  FUNCTIONS                                #
# ===================================================================== #

class SegmentationMetrics:
    """Accumulates predictions across batches and computes final metrics."""

    def __init__(self, num_classes=NUM_CLASSES, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion[:] = 0

    def update(self, pred, target):
        pred_np = pred.argmax(dim=1).cpu().numpy().ravel()
        tgt_np = target.cpu().numpy().ravel()
        valid = tgt_np != self.ignore_index
        pred_np = pred_np[valid]
        tgt_np = tgt_np[valid]
        np.add.at(self.confusion, (tgt_np, pred_np), 1)

    @property
    def pixel_accuracy(self):
        correct = np.diag(self.confusion).sum()
        total = self.confusion.sum()
        return correct / max(total, 1)

    @property
    def mean_accuracy(self):
        per_class = np.diag(self.confusion) / np.maximum(
            self.confusion.sum(axis=1), 1
        )
        return per_class.mean()

    @property
    def per_class_iou(self):
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / denom, 0.0)
        return iou

    @property
    def mean_iou(self):
        iou = self.per_class_iou
        valid = self.confusion.sum(axis=1) > 0
        return iou[valid].mean() if valid.any() else 0.0

    @property
    def frequency_weighted_iou(self):
        freq = self.confusion.sum(axis=1) / max(self.confusion.sum(), 1)
        iou = self.per_class_iou
        return (freq * iou).sum()

    def summary(self):
        lines = [
            f"Pixel Accuracy     : {self.pixel_accuracy:.4f}",
            f"Mean Accuracy      : {self.mean_accuracy:.4f}",
            f"Mean IoU           : {self.mean_iou:.4f}",
            f"Freq-Weighted IoU  : {self.frequency_weighted_iou:.4f}",
        ]
        for idx, name in enumerate(FLOORPLAN_CLASSES):
            lines.append(f"  {name:20s} IoU = {self.per_class_iou[idx]:.4f}")
        return "\n".join(lines)


class DepthMetrics:
    """Running depth-estimation metrics: RMSE, MAE, delta-thresholds."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._abs_rel = []
        self._sq_rel = []
        self._rmse = []
        self._mae = []
        self._delta1 = []
        self._delta2 = []
        self._delta3 = []

    def update(self, pred, target):
        pred_np = pred.detach().cpu().numpy().ravel()
        tgt_np = target.detach().cpu().numpy().ravel()
        valid = tgt_np > 1e-6
        p = pred_np[valid]
        t = tgt_np[valid]
        if len(t) == 0:
            return

        self._abs_rel.append(np.mean(np.abs(p - t) / t))
        self._sq_rel.append(np.mean(((p - t) ** 2) / t))
        self._rmse.append(np.sqrt(np.mean((p - t) ** 2)))
        self._mae.append(np.mean(np.abs(p - t)))

        ratio = np.maximum(p / t, t / p)
        self._delta1.append(np.mean(ratio < 1.25))
        self._delta2.append(np.mean(ratio < 1.25 ** 2))
        self._delta3.append(np.mean(ratio < 1.25 ** 3))

    def compute(self):
        def _avg(lst):
            return float(np.mean(lst)) if lst else 0.0
        return {
            "abs_rel": _avg(self._abs_rel),
            "sq_rel": _avg(self._sq_rel),
            "rmse": _avg(self._rmse),
            "mae": _avg(self._mae),
            "delta_1.25": _avg(self._delta1),
            "delta_1.25^2": _avg(self._delta2),
            "delta_1.25^3": _avg(self._delta3),
        }

    def summary(self):
        m = self.compute()
        return (
            f"Abs Rel: {m['abs_rel']:.4f}  Sq Rel: {m['sq_rel']:.4f}  "
            f"RMSE: {m['rmse']:.4f}  MAE: {m['mae']:.4f}\n"
            f"d<1.25: {m['delta_1.25']:.4f}  d<1.25^2: {m['delta_1.25^2']:.4f}  "
            f"d<1.25^3: {m['delta_1.25^3']:.4f}"
        )


# ===================================================================== #
#                        UTILITY  HELPERS                                #
# ===================================================================== #

def count_parameters(model):
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(3, 512, 512)):
    """Print a brief layer-by-layer summary."""
    device = next(model.parameters()).device
    dummy = torch.zeros(1, *input_size, device=device)
    total_params = count_parameters(model)
    print(f"{'Layer':<45} {'Output Shape':<25} {'Params':>12}")
    print("-" * 85)
    hooks = []
    info = []

    def _hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                shapes = [str(list(o.shape)) for o in out if isinstance(o, torch.Tensor)]
                shape_str = ", ".join(shapes)
            else:
                shape_str = str(list(out.shape))
            n_params = sum(p.numel() for p in module.parameters(recurse=False))
            info.append((name, shape_str, n_params))
        return fn

    for name, module in model.named_modules():
        if name and not any(isinstance(module, t) for t in (nn.Sequential, nn.ModuleList)):
            hooks.append(module.register_forward_hook(_hook(name)))

    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    for name, shape, n in info:
        print(f"{name:<45} {shape:<25} {n:>12,}")

    print("-" * 85)
    print(f"Total trainable parameters: {total_params:,}")


def load_model(path, model_class=MultiTaskFloorplanNet, device=None, **kwargs):
    """Load a saved checkpoint into *model_class* and return model + metadata."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = model_class(**kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def save_model(path, model, optimizer, epoch, metrics):
    """Persist model weights and training metadata."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def export_to_onnx(model, output_path, input_size=(1, 3, 512, 512)):
    """Export the model to ONNX format for deployment."""
    device = next(model.parameters()).device
    dummy = torch.randn(*input_size, device=device)
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=13,
        input_names=["input"],
        output_names=["segmentation", "depth"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "segmentation": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 2: "height", 3: "width"},
        },
    )


def freeze_encoder(model):
    """Freeze all encoder parameters for transfer-learning fine-tuning."""
    for name, param in model.named_parameters():
        if name.startswith("enc") or name.startswith("bottleneck"):
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze every parameter in the model."""
    for param in model.parameters():
        param.requires_grad = True


def get_learning_rate_groups(model, encoder_lr=1e-4, decoder_lr=1e-3):
    """Return per-layer parameter groups with different learning rates.

    Allows fine-tuning the encoder more slowly than the decoder, which is
    especially helpful after pre-training on a larger dataset.
    """
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("enc") or name.startswith("bottleneck"):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    return [
        {"params": encoder_params, "lr": encoder_lr},
        {"params": decoder_params, "lr": decoder_lr},
    ]
