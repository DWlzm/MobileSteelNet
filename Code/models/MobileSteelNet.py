"""
MobileSteelNet: A Lightweight Steel Surface Defect Classification Network
with Cross-Interactive Efficient Multi-Scale Attention (CIEMA)

Reference:
    Zou, X.; Liu, Z.; Xu, C.; Zhang, J.; Li, Z.
    MobileSteelNet: A Lightweight Steel Surface Defect Classification Network
    with Cross-Interactive Efficient Multi-Scale Attention. Sensors 2026, 26, 1022.
    https://www.mdpi.com/1424-8220/26/3/1022

This file is a self-contained implementation and does NOT depend on other modules
in the models/ folder (EMA.py / MultiScaleFusion.py / EMA_enhanced.py, etc.).
All core modules (DepthwiseSeparableConv, MSFF, CIEMA) are fully defined in this file.

Key points of the paper:
    1. MSFF (Multi-Scale Feature Fusion): fuses multi-stage features.
    2. CIEMA (Cross-Interactive Efficient Multi-Scale Attention):
       unifies inter-channel interaction, parallel multi-scale spatial extraction,
       and grouped efficient computation.
    3. Model size is only 8.2 MB; average accuracy of 91.36% on NEU-DET.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Base convolution block: Depthwise Separable Convolution
# ============================================================================
class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


# ============================================================================
# MSFF: Multi-Scale Feature Fusion module
# ============================================================================
class MSFF(nn.Module):
    """
    Multi-Scale Feature Fusion (MSFF)
    Takes multi-scale feature maps from different stages, unifies their spatial sizes,
    concatenates along the channel dimension, and fuses with a 1x1 convolution.
    """

    def __init__(self, stage_channels, out_channels=None):
        super(MSFF, self).__init__()
        self.stage_channels = list(stage_channels)
        self.num_stages = len(self.stage_channels)

        if out_channels is None:
            out_channels = max(self.stage_channels)
        self.out_channels = out_channels

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(self.stage_channels), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, stage_features):
        # Unify to the maximum spatial size
        max_h, max_w = 0, 0
        for feat in stage_features:
            max_h = max(max_h, feat.shape[2])
            max_w = max(max_w, feat.shape[3])

        resized_features = []
        for feat in stage_features:
            if feat.shape[2] != max_h or feat.shape[3] != max_w:
                feat = F.interpolate(feat, size=(max_h, max_w), mode='bilinear', align_corners=False)
            resized_features.append(feat)

        fused = torch.cat(resized_features, dim=1)
        return self.fusion_conv(fused)


# ============================================================================
# CIEMA: Cross-Interactive Efficient Multi-Scale Attention
# ============================================================================
class ChannelInteraction(nn.Module):
    """Inter-channel interaction module (one of the CIEMA sub-modules)"""

    def __init__(self, channels, reduction=16):
        super(ChannelInteraction, self).__init__()
        self.channels = channels
        self.reduction = reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False)
        )

        self.channel_interaction = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)

        avg_att = self.fc(avg_out)
        max_att = self.fc(max_out)
        channel_att = torch.sigmoid(avg_att + max_att).view(b, c, 1, 1)

        channel_interaction = self.channel_interaction(x)

        enhanced_x = x * channel_att + channel_interaction * (1 - channel_att)
        return enhanced_x


class CrossSpatialLearning(nn.Module):
    """Enhanced cross-spatial learning module (one of the CIEMA sub-modules):
    parallel multi-scale spatial extraction + channel modeling."""

    def __init__(self, channels, groups=32):
        super(CrossSpatialLearning, self).__init__()
        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        assert self.group_channels > 0, "channels must be divisible by groups"

        # Multi-scale spatial feature extraction
        self.spatial_conv1 = nn.Conv2d(self.group_channels, self.group_channels, 1, bias=False)
        self.spatial_conv3 = nn.Conv2d(self.group_channels, self.group_channels, 3, padding=1, bias=False)
        self.spatial_conv5 = nn.Conv2d(self.group_channels, self.group_channels, 5, padding=2, bias=False)

        # Spatial attention weight generation
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.group_channels * 3, self.group_channels * 3, 1, bias=False),
            nn.BatchNorm2d(self.group_channels * 3),
            nn.Sigmoid()
        )

        # Cross-spatial interaction matrix
        self.channel_feat = nn.Conv2d(self.group_channels, self.group_channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))  # Adaptive weighting factor

        # Compress 3x channels of multi-scale features back to per-group channels
        self.ms_reduce = nn.Sequential(
            nn.Conv2d(self.group_channels * 3, self.group_channels, 1, bias=False),
            nn.BatchNorm2d(self.group_channels)
        )

    def forward(self, x):
        # Multi-scale spatial features
        feat1 = self.spatial_conv1(x)
        feat3 = self.spatial_conv3(x)
        feat5 = self.spatial_conv5(x)

        multi_scale_feat = torch.cat([feat1, feat3, feat5], dim=1)
        spatial_att = self.spatial_attention(multi_scale_feat)
        multi_scale_feat = spatial_att * multi_scale_feat
        multi_scale_feat = self.ms_reduce(multi_scale_feat)

        # Cross-spatial interaction
        channel_feat = self.channel_feat(x)
        channel_weighted_feat = channel_feat * self.alpha

        enhanced_x = multi_scale_feat + channel_weighted_feat
        return enhanced_x


class CIEMA(nn.Module):
    """
    Cross-Interactive Efficient Multi-Scale Attention (CIEMA)
    The core attention module of the paper, unifying inter-channel interaction,
    parallel multi-scale spatial extraction, and grouped efficient computation.
    """

    def __init__(self, channels, c2=None, factor=32, reduction=16):
        super(CIEMA, self).__init__()
        self.groups = factor
        self.channels = channels
        assert channels // self.groups > 0, "channels must be greater than groups"

        # Coordinate attention (preserve the original EMA logic)
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(
            channels // self.groups, channels // self.groups,
            kernel_size=1, stride=1, padding=0
        )
        self.conv3x3 = nn.Conv2d(
            channels // self.groups, channels // self.groups,
            kernel_size=3, stride=1, padding=1
        )

        # Inter-channel interaction module
        self.channel_interaction = ChannelInteraction(channels, reduction)

        # Enhanced cross-spatial learning module
        self.cross_spatial_learning = CrossSpatialLearning(channels, self.groups)

        # Multi-scale feature fusion
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (B, C, H, W)
        b, c, h, w = x.size()

        # 1. Inter-channel interaction enhancement
        channel_enhanced_x = self.channel_interaction(x)

        # 2. Coordinate attention module (preserve the original EMA logic)
        group_x = channel_enhanced_x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 1x1 branch and 3x3 branch
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        # Enhanced cross-spatial learning: process each group separately
        x1_reshaped = x1.reshape(b, c, h, w)
        x2_reshaped = x2.reshape(b, c, h, w)

        x1_enhanced_list = []
        x2_enhanced_list = []
        for i in range(self.groups):
            start_ch = i * (c // self.groups)
            end_ch = (i + 1) * (c // self.groups)

            x1_group = x1_reshaped[:, start_ch:end_ch, :, :]
            x2_group = x2_reshaped[:, start_ch:end_ch, :, :]

            x1_enhanced_group = self.cross_spatial_learning(x1_group)
            x2_enhanced_group = self.cross_spatial_learning(x2_group)

            x1_enhanced_list.append(x1_enhanced_group)
            x2_enhanced_list.append(x2_enhanced_group)

        x1_enhanced = torch.cat(x1_enhanced_list, dim=1).reshape(b * self.groups, -1, h, w)
        x2_enhanced = torch.cat(x2_enhanced_list, dim=1).reshape(b * self.groups, -1, h, w)

        # 3. Enhanced cross-spatial learning (cross-adjustment with channel descriptors)
        x11 = self.softmax(
            self.agp(x1_enhanced).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )
        x12 = x2_enhanced.reshape(b * self.groups, c // self.groups, -1)
        y1 = torch.matmul(x11, x12)

        x21 = self.softmax(
            self.agp(x2_enhanced).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )
        x22 = x1_enhanced.reshape(b * self.groups, c // self.groups, -1)
        y2 = torch.matmul(x21, x22)

        # 4. Spatial weight generation
        weights = (y1 + y2).reshape(b * self.groups, 1, h, w)
        weights_ = weights.sigmoid()
        spatial_enhanced = (group_x * weights_).reshape(b, c, h, w)

        # 5. Multi-scale feature fusion
        multi_scale_feat = self.multi_scale_fusion(spatial_enhanced)

        # 6. Final fusion
        combined_feat = torch.cat([spatial_enhanced, multi_scale_feat], dim=1)
        final_weights = self.final_fusion(combined_feat)

        # 7. Output
        out = channel_enhanced_x * final_weights + spatial_enhanced * (1 - final_weights)
        return out


# ============================================================================
# MobileSteelNet: Main network
# ============================================================================
class MobileSteelNet(nn.Module):
    """
    MobileSteelNet main network

    Architecture:
        Stem -> Stage1 (shallow) -> Stage2 (middle) -> Stage3 (deep)
              -> MSFF (fuse stage1/2/3 features) -> CIEMA attention -> GAP -> classifier

    Paper reports: model size 8.2 MB, average accuracy of 91.36% on NEU-DET.
    """

    def __init__(self, num_classes=6, in_channels=3):
        super(MobileSteelNet, self).__init__()

        # Initial convolution layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 1: shallow feature extraction
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1)
        )

        # Stage 2: middle-level feature extraction
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1)
        )

        # Stage 3: deep feature extraction
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1)
        )

        # Multi-Scale Feature Fusion (MSFF)
        self.msff = MSFF([128, 512, 1024], 1024)

        # CIEMA attention
        self.ciema = CIEMA(1024, factor=32, reduction=16)

        # Classification head
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Stem
        x = self.initial_conv(x)

        # Three-stage feature extraction
        stage1_feat = self.stage1(x)   # 128 channels
        stage2_feat = self.stage2(stage1_feat)  # 512 channels
        stage3_feat = self.stage3(stage2_feat)  # 1024 channels

        # MSFF multi-scale feature fusion
        fused = self.msff([stage1_feat, stage2_feat, stage3_feat])

        # CIEMA attention enhancement
        enhanced = self.ciema(fused)

        # Classification
        out = self.adaptive_pool(enhanced)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ============================================================================
# Tensor tests
# ============================================================================
def _test_ciema():
    """Test the CIEMA attention module"""
    print("=" * 60)
    print("[Test 1] CIEMA attention module")
    print("-" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(2, 1024, 7, 7, device=device)
    model = CIEMA(channels=1024, factor=32, reduction=16).to(device)

    out = model(x)
    print(f"  Input  tensor shape: {tuple(x.shape)}")
    print(f"  Output tensor shape: {tuple(out.shape)}")
    assert out.shape == x.shape, f"CIEMA output shape should equal input shape, got {out.shape}"
    print("  [OK] CIEMA input/output shape match")

    # Backward pass test
    loss = out.sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grad_ok, "CIEMA backward failed: some parameters have no gradient"
    print("  [OK] CIEMA backward pass works")
    print()


def _test_msff():
    """Test the MSFF multi-scale feature fusion module"""
    print("=" * 60)
    print("[Test 2] MSFF multi-scale feature fusion module")
    print("-" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    s1 = torch.randn(2, 128, 56, 56, device=device)
    s2 = torch.randn(2, 512, 28, 28, device=device)
    s3 = torch.randn(2, 1024, 14, 14, device=device)

    model = MSFF([128, 512, 1024], 1024).to(device)
    out = model([s1, s2, s3])
    print(f"  Input stage1 shape: {tuple(s1.shape)}")
    print(f"  Input stage2 shape: {tuple(s2.shape)}")
    print(f"  Input stage3 shape: {tuple(s3.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    assert out.shape[1] == 1024, "MSFF output channels should be 1024"
    print("  [OK] MSFF multi-scale fusion shape is correct")

    loss = out.sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grad_ok, "MSFF backward failed"
    print("  [OK] MSFF backward pass works")
    print()


def _test_mobilesteelnet():
    """Test the full MobileSteelNet network"""
    print("=" * 60)
    print("[Test 3] MobileSteelNet full network")
    print("-" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MobileSteelNet(num_classes=6).to(device)
    x = torch.randn(4, 3, 224, 224, device=device)

    out = model(x)
    print(f"  Input  tensor shape: {tuple(x.shape)}")
    print(f"  Output tensor shape: {tuple(out.shape)}")
    assert out.shape == (4, 6), f"MobileSteelNet output shape should be (4, 6), got {out.shape}"
    print("  [OK] Output shape is correct: (B, num_classes)")

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Approx. {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    # Backward pass
    target = torch.randint(0, 6, (4,), device=device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, target)
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grad_ok, "MobileSteelNet backward failed"
    print(f"  [OK] Backward pass works, loss = {loss.item():.4f}")
    print()


def _test_dynamic_input():
    """Test robustness under different input sizes"""
    print("=" * 60)
    print("[Test 4] Robustness under different input sizes")
    print("-" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileSteelNet(num_classes=6).to(device)
    model.eval()

    for size in [160, 224, 256]:
        x = torch.randn(1, 3, size, size, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 6), f"Input {size}x{size} produced wrong output shape: {out.shape}"
        print(f"  [OK] Input {size}x{size} -> Output {tuple(out.shape)}")
    print()


if __name__ == '__main__':
    print("\nMobileSteelNet tensor tests")
    print("Paper: MobileSteelNet: A Lightweight Steel Surface Defect "
          "Classification Network with Cross-Interactive Efficient Multi-Scale Attention")
    print()

    _test_ciema()
    _test_msff()
    _test_mobilesteelnet()
    _test_dynamic_input()

    print("=" * 60)
    print("All tensor tests passed!")
    print("=" * 60)
