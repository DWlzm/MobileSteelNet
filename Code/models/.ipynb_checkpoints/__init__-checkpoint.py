# 模型包初始化文件
from .resnet import ResNet18, ResNet34, ResNet50
from .mobilenet import MobileNetV1, MobileNetV2, MobileNetV3
from .shufflenet import ShuffleNetV1, ShuffleNetV2
from .few_shot import PrototypicalNetworks, MAML, FewShotTrainer
from .EMA import EMA

__all__ = [
    'ResNet18', 'ResNet34', 'ResNet50',
    'MobileNetV1', 'MobileNetV2', 'MobileNetV3',
    'ShuffleNetV1', 'ShuffleNetV2',
    'PrototypicalNetworks', 'MAML', 'FewShotTrainer',
    'EMA'
]
