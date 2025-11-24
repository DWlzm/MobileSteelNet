import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # EnhancedEMA特征融合
        # self.enhanced_ema = EnhancedEMA(out_channels)
        
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
        
        # EnhancedEMA特征融合
        # enhanced_features = self.enhanced_ema(fused_features)
        
        return fused_features


class ChannelInteraction(nn.Module):
    """通道间交互模块"""
    def __init__(self, channels, reduction=16):
        super(ChannelInteraction, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 通道注意力分支
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # 通道间交互矩阵
        self.channel_interaction = nn.Conv2d(channels, channels, 1, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 全局特征提取
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # 通道注意力
        avg_att = self.fc(avg_out)
        max_att = self.fc(max_out)
        channel_att = torch.sigmoid(avg_att + max_att).view(b, c, 1, 1)
        
        # 通道间交互
        channel_interaction = self.channel_interaction(x)
        
        # 结合通道注意力和通道间交互
        enhanced_x = x * channel_att + channel_interaction * (1 - channel_att)
        
        return enhanced_x

class CrossSpatialLearning(nn.Module):
    """增强的跨空间学习模块"""
    def __init__(self, channels, groups=32):
        super(CrossSpatialLearning, self).__init__()
        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        
        # 多尺度空间特征提取
        self.spatial_conv1 = nn.Conv2d(self.group_channels, self.group_channels, 1, bias=False)
        self.spatial_conv3 = nn.Conv2d(self.group_channels, self.group_channels, 3, padding=1, bias=False)
        self.spatial_conv5 = nn.Conv2d(self.group_channels, self.group_channels, 5, padding=2, bias=False)
        
        # 空间注意力权重生成
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.group_channels * 3, self.group_channels, 1, bias=False),
            nn.BatchNorm2d(self.group_channels),
            nn.Sigmoid()
        )
        
        # 跨空间交互矩阵
        self.cross_spatial_conv = nn.Conv2d(self.group_channels, self.group_channels, 1, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 多尺度空间特征
        feat1 = self.spatial_conv1(x)
        feat3 = self.spatial_conv3(x)
        feat5 = self.spatial_conv5(x)
        
        # 多尺度特征融合
        multi_scale_feat = torch.cat([feat1, feat3, feat5], dim=1)
        spatial_att = self.spatial_attention(multi_scale_feat)
        
        # 跨空间交互
        cross_spatial = self.cross_spatial_conv(x)
        
        # 结合空间注意力和跨空间交互
        enhanced_x = x * spatial_att + cross_spatial * (1 - spatial_att)
        
        return enhanced_x

class EnhancedEMA(nn.Module):
    """增强版EMA注意力模块"""
    def __init__(self, channels, c2=None, factor=32, reduction=16):
        super(EnhancedEMA, self).__init__()
        self.groups = factor
        self.channels = channels
        assert channels // self.groups > 0
        
        # 坐标注意力模块（保持原有功能）
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        
        # 新增：通道间交互模块
        self.channel_interaction = ChannelInteraction(channels, reduction)
        
        # 新增：增强的跨空间学习模块
        self.cross_spatial_learning = CrossSpatialLearning(channels, self.groups)
        
        # 新增：多尺度特征融合
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # 新增：最终融合层
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        #(B,C,H,W)
        b, c, h, w = x.size()
        
        # 1. 通道间交互增强
        channel_enhanced_x = self.channel_interaction(x)
        
        # 2. 坐标注意力模块（保持原有逻辑）
        group_x = channel_enhanced_x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        
        # 1×1分支和3×3分支
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        
        # 跨空间学习（增强版）
        x1_reshaped = x1.reshape(b, c, h, w)
        x2_reshaped = x2.reshape(b, c, h, w)
        
        # 对每个组分别处理
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
        
        # 3. 增强的跨空间学习
        x11 = self.softmax(self.agp(x1_enhanced).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2_enhanced.reshape(b * self.groups, c // self.groups, -1)
        y1 = torch.matmul(x11, x12)
        
        x21 = self.softmax(self.agp(x2_enhanced).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1_enhanced.reshape(b * self.groups, c // self.groups, -1)
        y2 = torch.matmul(x21, x22)
        
        # 4. 多尺度特征融合
        weights = (y1 + y2).reshape(b * self.groups, 1, h, w)
        weights_ = weights.sigmoid()
        spatial_enhanced = (group_x * weights_).reshape(b, c, h, w)
        
        # 5. 多尺度特征融合
        multi_scale_feat = self.multi_scale_fusion(spatial_enhanced)
        
        # 6. 最终融合
        combined_feat = torch.cat([spatial_enhanced, multi_scale_feat], dim=1)
        final_weights = self.final_fusion(combined_feat)
        
        # 7. 输出
        out = channel_enhanced_x * final_weights + spatial_enhanced * (1 - final_weights)
        
        return out
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                                 padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=6):
        super(MobileNetV1, self).__init__()
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 第一阶段：浅层特征提取
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1)
        )
        
        # 第二阶段：中层特征提取
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1)
        )
        
        # 第三阶段：深层特征提取
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 512, 1),
            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1)
        )
        
        # 多尺度融合模块 - 融合来自不同stage的特征
        self.multiscale_fusion = MultiScaleFusion([128, 512, 1024], 1024)
        
        self.enhanced_ema = EnhancedEMA(1024)
        # self.ema = EMA(1024)
        
        # self.se = SEBlock(1024)
#         self.ca = ChannelAttention(self.last_channel)
#         # self.sa = SpatialAttention(self.last_channel)bug
        # self.cbam = CBAM(1024)
        # self.eca = ECABlock(1024)
        # self.CoordAtt = CoordAtt(1024, 1024)
        
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 初始特征提取
        x = self.initial_conv(x)
        
        # 第一阶段：浅层特征提取
        stage1_feat = self.stage1(x)
        
        # 第二阶段：中层特征提取
        stage2_feat = self.stage2(stage1_feat)
        
        # 第三阶段：深层特征提取
        stage3_feat = self.stage3(stage2_feat)
        
        # x=stage3_feat
        # 多尺度特征融合
        fused_features = self.multiscale_fusion([stage1_feat, stage2_feat, stage3_feat])
        # x=fused_features
        
        # x = self.se(fused_features)
        # x = self.cbam(fused_features)
        # x = self.ca(x)
        # x = self.sa(x)
        # x = self.cbam(x)
        # x = self.eca(fused_features)
        # x=self.CoordAtt(fused_features)
        # x = self.ema(fused_features)
        # 增强EMA和分类
        x = self.enhanced_ema(fused_features)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        



if __name__ == '__main__':
    # (B,C,H,W)
    input=torch.randn(1,3,200,200)
    Model = MobileNetV1()
    output=Model(input)
    print(output.shape)
    print(Model)


