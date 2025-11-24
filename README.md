# MobileSteelNet 钢铁表面缺陷图像分类项目

### 代码目标结构
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


### 数据集 NEU-DET

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
    
    
    


    
