# MobileSteelNet 钢铁表面缺陷图像分类项目

### 代码目标结构
```
Code/
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── resnet.py          # ResNet系列模型
│   ├── mobilenet.py       # MobileNet系列模型
│   └── shufflenet.py      # ShuffleNet系列模型
├── dataset/               # 数据集处理
│   ├── __init__.py
│   ├── steel_dataset.py   # 数据集类
│   └── transforms.py      # 数据变换
├── utils/                  # 工具类
│   ├── __init__.py
│   ├── metrics.py         # 评估指标
│   └── visualization.py   # 可视化
├── train.py               # 训练脚本
├── config.py              # 配置文件
├── requirements.txt       # 依赖包
└── README.md             # 说明文档
```
## 支持的模型

- **ResNet系列**: ResNet18, ResNet34, ResNet50
- **MobileNet系列**: MobileNetV1, MobileNetV2, MobileNetV3
- **ShuffleNet系列**: ShuffleNetV1, ShuffleNetV2


### 数据集 NEU-DET

```
Data/
├── NEU-DET.zip                # 原始压缩包（可选，程序支持自动解压）
│
├── NEU-DET/                   # 解压后的NEU-DET数据集目录
│   ├── crazing/               # 裂纹缺陷图像
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...（共300张图像）
│   ├── inclusion/             # 夹杂缺陷图像
│   │   ├── 1.png
│   │   └── ...（共300张图像）
│   ├── patches/               # 斑块缺陷图像
│   ├── pitted/                # 点蚀缺陷图像
│   ├── rolled-in/             # 轧制氧化皮缺陷图像
│   └── scratches/             # 划痕缺陷图像
│
├── FSSD-12.zip                # 原始压缩包（可选）
│
└── FSSD-12/                   # 解压后的FSSD-12数据集目录
    ├── crazing/               # 裂纹缺陷图像
    ├── inclusion/             # 夹杂缺陷图像
    ├── patches/               # 斑块缺陷图像
    ├── pitted/                # 点蚀缺陷图像
    ├── rolled-in/             # 轧制氧化皮缺陷图像
    ├── scratches/             # 划痕缺陷图像
    ├── ...（额外6类缺陷目录，共12类）
```    
    
    
## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将数据集放在 `Data/` 目录下：

```
Data/
├── NEU-DET.zip
└── FSSD-12.zip
```

### 3. 训练模型

```python
# 使用默认配置训练
python train.py

# 或者修改配置后训练
from train import SteelDefectTrainer
from config import get_config

config = get_config(dataset_type='NEU-DET', model_name='ResNet18')
trainer = SteelDefectTrainer(config)
trainer.train(train_loader, val_loader, test_loader)
```

### 4. 配置参数

在 `config.py` 中可以修改以下参数：

- **数据集配置**: 数据集类型、类别数量、图像大小
- **模型配置**: 模型类型、预训练权重
- **训练配置**: 学习率、批次大小、训练轮次
- **数据增强**: 是否使用增强、增强参数
- **保存配置**: 保存路径、保存间隔

## 示例配置

```python
config = {
    'data_dir': '/path/to/data',
    'dataset_type': 'NEU-DET',
    'model_name': 'ResNet18',
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'scheduler': 'StepLR'
}
```

## 输出结果

训练完成后会生成以下文件：

- `checkpoints/`: 模型检查点
- `results/`: 评估结果和可视化图表
  - `metrics.txt`: 详细评估指标
  - `confusion_matrix.png`: 混淆矩阵
  - `training_curves.png`: 训练曲线
  - `class_distribution.png`: 类别分布


## CIEMA
```python

import torch
import torch.nn as nn
import torch.nn.functional as F

"Enhanced Efficient Multi-Scale Attention Module with Cross-Spatial Learning and Channel Interaction"

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
            nn.Conv2d(self.group_channels * 3, self.group_channels * 3, 1, bias=False),
            nn.BatchNorm2d(self.group_channels * 3),
            nn.Sigmoid()
        )
        
        # 跨空间交互矩阵
        self.channel_feat = nn.Conv2d(self.group_channels, self.group_channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))  # 自适应调整参数，用于加权通道建模

        # 将3倍通道的多尺度特征压缩回每组通道数，便于与通道建模结果相加
        self.ms_reduce = nn.Sequential(
            nn.Conv2d(self.group_channels * 3, self.group_channels, 1, bias=False),
            nn.BatchNorm2d(self.group_channels)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 多尺度空间特征
        feat1 = self.spatial_conv1(x)
        feat3 = self.spatial_conv3(x)
        feat5 = self.spatial_conv5(x)
        
        # 多尺度特征融合
        multi_scale_feat = torch.cat([feat1, feat3, feat5], dim=1)
        spatial_att = self.spatial_attention(multi_scale_feat)
        multi_scale_feat=spatial_att*multi_scale_feat
        multi_scale_feat=self.ms_reduce(multi_scale_feat)
        
        # 跨空间交互
        channel_feat = self.channel_feat(x)
        
        # 使用alpha对通道建模结果进行加权
        channel_weighted_feat = channel_feat * self.alpha
        
        # 结合空间注意力和跨空间交互
        enhanced_x =multi_scale_feat+ channel_weighted_feat
        
        return enhanced_x
        # return x  

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
        # out = channel_enhanced_x+multi_scale_feat
        return out

```



## 注意事项

1. 确保有足够的GPU内存进行训练
2. 根据数据集大小调整批次大小
3. 可以根据需要调整学习率和训练轮次
4. 建议使用数据增强来提高模型泛化能力



    
