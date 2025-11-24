import torch
import torch.nn as nn
import torch.nn.functional as F
from .EMA import EMA
from .Attention import SEBlock , ChannelAttention, SpatialAttention, CBAM, ECABlock, CoordAtt, MultiHeadSelfAttention, AttentionGate, PyramidAttention
from .EMA_enhanced import EnhancedEMA
from .MultiScaleFusion import MultiScaleFusion, AdaptiveMultiScaleFusion

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
        






class MobileNetV2(nn.Module):
    def __init__(self, num_classes=6, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                   nn.BatchNorm2d(input_channel),
                   nn.ReLU6(inplace=True)]
        
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        
        features.append(nn.Conv2d(input_channel, self.last_channel, 1, bias=False))
        features.append(nn.BatchNorm2d(self.last_channel))
        features.append(nn.ReLU6(inplace=True))
        
        self.features = nn.Sequential(*features)
        # self.ema = EMA(self.last_channel)
        # self.se = SEBlock(self.last_channel)
        # self.ca = ChannelAttention(self.last_channel)
        # self.sa = SpatialAttention(self.last_channel)bug
        # self.cbam = CBAM(self.last_channel)
        # self.eca = ECABlock(self.last_channel)
        # self.CoordAtt = CoordAtt(self.last_channel, self.last_channel)
        self.classifier = nn.Linear(self.last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        # x = self.ema(x)
        # x = self.se(x)
        # x = self.ca(x)
        # x = self.sa(x)
        # x = self.cbam(x)
        # x = self.eca(x)
        # x=self.CoordAtt(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=6, width_mult=1.0):
        super(MobileNetV3, self).__init__()
        self.cfgs = [
            [3, 16, 16, False, 'RE', 1, 1],
            [3, 64, 24, False, 'RE', 2, 1],
            [3, 72, 24, False, 'RE', 1, 1],
            [5, 72, 40, True, 'RE', 2, 1],
            [5, 120, 40, True, 'RE', 1, 1],
            [5, 120, 40, True, 'RE', 1, 1],
            [3, 240, 80, False, 'HS', 2, 1],
            [3, 200, 80, False, 'HS', 1, 1],
            [3, 184, 80, False, 'HS', 1, 1],
            [3, 184, 80, False, 'HS', 1, 1],
            [3, 480, 112, True, 'HS', 1, 1],
            [3, 672, 112, True, 'HS', 1, 1],
            [5, 672, 160, True, 'HS', 2, 1],
            [5, 960, 160, True, 'HS', 1, 1],
            [5, 960, 160, True, 'HS', 1, 1],
        ]
        
        input_channel = 16
        last_channel = 1280
        
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult)
        
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                   nn.BatchNorm2d(input_channel),
                   nn.Hardswish(inplace=True)]
        
        for k, exp, c, se, act, s, e in self.cfgs:
            output_channel = int(c * width_mult)
            exp_channel = int(exp * width_mult)
            features.append(InvertedResidual(input_channel, output_channel, s, e))
            input_channel = output_channel
        
        features.append(nn.Conv2d(input_channel, self.last_channel, 1, bias=False))
        features.append(nn.BatchNorm2d(self.last_channel))
        features.append(nn.Hardswish(inplace=True))
        
        self.features = nn.Sequential(*features)
        self.ema = EMA(self.last_channel)
        self.classifier = nn.Sequential(
            nn.Linear(self.last_channel, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.ema(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

