import torch
import torch.nn as nn
import torch.nn.functional as F
from .EMA_enhanced import EnhancedEMA



class MultiScaleFusion(nn.Module):
    """
    多尺度特征融合模块
    接收来自不同stage的多尺度特征图，进行融合
    """
    def __init__(self, stage_channels, out_channels=None):
        super(MultiScaleFusion, self).__init__()
        # stage_channels: [stage1_channels, stage2_channels, stage3_channels]
        self.stage_channels = stage_channels
        self.num_stages = len(stage_channels)
        
        if out_channels is None:
            out_channels = max(stage_channels)
        self.out_channels = out_channels
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(stage_channels), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, stage_features):
        """
        Args:
            stage_features: list of tensors from different stages
        Returns:
            fused features
        """
        # 将所有特征图调整到相同尺寸（使用最大尺寸）
        max_h, max_w = 0, 0
        for feat in stage_features:
            max_h = max(max_h, feat.shape[2])
            max_w = max(max_w, feat.shape[3])
        
        # 调整所有特征图到相同尺寸
        resized_features = []
        for feat in stage_features:
            if feat.shape[2] != max_h or feat.shape[3] != max_w:
                feat = F.interpolate(feat, size=(max_h, max_w), mode='bilinear', align_corners=False)
            resized_features.append(feat)
        
        # 通道叠加
        fused_features = torch.cat(resized_features, dim=1)
        # 特征融合
        fused_features = self.fusion_conv(fused_features)
        return fused_features
