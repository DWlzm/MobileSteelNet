# 钢铁表面缺陷图像分类

这是一个用于钢铁表面缺陷图像分类的深度学习项目，支持多种模型架构和数据集。

## 项目结构

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

## 支持的数据集

- **NEU-DET**: 6个类别的钢铁表面缺陷数据集
- **FSSD-12**: 12个类别的钢铁表面缺陷数据集

## 功能特性

### 1. 模型训练
- 支持多种深度学习模型
- 自动数据加载和预处理
- 训练过程可视化
- 模型检查点保存

### 2. 评估指标
- 混淆矩阵
- 各类别准确率
- 总准确率
- 精确率、召回率、F1分数

### 3. 数据增强
- 基础数据增强（旋转、翻转、颜色变换等）
- 高级数据增强（MixUp、CutMix等）
- 测试时增强（TTA）

### 4. 可视化
- 训练曲线
- 混淆矩阵热力图
- 类别分布图
- 学习率调度图

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

## 注意事项

1. 确保有足够的GPU内存进行训练
2. 根据数据集大小调整批次大小
3. 可以根据需要调整学习率和训练轮次
4. 建议使用数据增强来提高模型泛化能力

## 扩展功能

- 支持自定义模型架构
- 支持更多数据增强技术
- 支持模型集成和投票
- 支持在线学习和增量学习

