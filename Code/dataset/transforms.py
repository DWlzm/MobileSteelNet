import torch
from torchvision import transforms
import random

def get_transforms(image_size=224, augmentation=True):
    """
    获取数据变换
    
    Args:
        image_size (int): 图像大小
        augmentation (bool): 是否使用数据增强
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    
    if augmentation:
        # 训练集变换（包含数据增强）
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # 训练集变换（无数据增强）
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 验证集变换
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_advanced_transforms(image_size=224):
    """
    获取高级数据变换（包含更多增强技术）
    
    Args:
        image_size (int): 图像大小
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    
    # 高级训练集变换
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集变换
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_test_time_augmentation(image_size=224):
    """
    获取测试时增强变换
    
    Args:
        image_size (int): 图像大小
    
    Returns:
        list: 测试时增强变换列表
    """
    
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 测试时增强变换
    tta_transforms = [
        # 原始图像
        base_transform,
        # 水平翻转
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # 垂直翻转
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # 旋转90度
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # 旋转180度
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # 旋转270度
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=270),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]
    
    return tta_transforms

class MixUp:
    """MixUp数据增强"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        
        # 生成混合权重
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        
        # 随机打乱索引
        indices = torch.randperm(batch_size)
        
        # 混合图像
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # 混合标签
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels

class CutMix:
    """CutMix数据增强"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        batch_size, channels, height, width = images.size()
        
        # 生成混合权重
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        
        # 随机打乱索引
        indices = torch.randperm(batch_size)
        
        # 计算裁剪区域
        cut_rat = torch.sqrt(1. - lam)
        cut_w = int(width * cut_rat)
        cut_h = int(height * cut_rat)
        
        # 随机选择裁剪中心
        cx = torch.randint(0, width, (1,))
        cy = torch.randint(0, height, (1,))
        
        # 计算裁剪边界
        bbx1 = torch.clamp(cx - cut_w // 2, 0, width)
        bby1 = torch.clamp(cy - cut_h // 2, 0, height)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, width)
        bby2 = torch.clamp(cy + cut_h // 2, 0, height)
        
        # 应用CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))
        
        # 混合标签
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels

