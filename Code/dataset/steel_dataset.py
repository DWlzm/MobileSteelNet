import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from torchvision import datasets as tv_datasets

class SteelDefectDataset(Dataset):
    """
    钢铁表面缺陷数据集类
    支持NEU-DET、FSSD-12 与 CIFAR-100 数据集
    """
    
    def __init__(self, data_dir, dataset_type='NEU-DET', split='train', transform=None):
        """
        初始化数据集
        
        Args:
            data_dir (str): 数据根目录
            dataset_type (str): 数据集类型 ('NEU-DET' 或 'FSSD-12')
            split (str): 数据集分割 ('train', 'val', 'test')
            transform: 数据变换
        """
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.split = split
        self.transform = transform
        
        # 数据集类别映射
        if dataset_type == 'NEU-DET':
            self.class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface', 'Rolled_in_scale', 'Scratches']
            self.folder_names = ['crazing', 'inclusion', 'patches', 'pitted', 'rolled-in', 'scratches']
            self.num_classes = 6
        elif dataset_type == 'FSSD-12':
            self.class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface', 'Rolled_in_scale', 'Scratches',
                              'Rolled_in_scale', 'Patches', 'Inclusion', 'Crazing', 'Pitted_surface', 'Scratches']
            self.folder_names = ['crazing', 'inclusion', 'patches', 'pitted', 'rolled-in', 'scratches',
                               'rolled-in', 'patches', 'inclusion', 'crazing', 'pitted', 'scratches']
            self.num_classes = 12
        elif dataset_type == 'CIFAR-100':
            # CIFAR-100 使用torchvision直接下载并读取
            # 先加载训练集以获取类别名
            cifar_train = tv_datasets.CIFAR100(root=self.data_dir, train=True, download=True)
            self.class_names = list(cifar_train.classes)
            self.folder_names = None
            self.num_classes = 100
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        # 解压数据集（如果需要）
        self._extract_dataset()
        
        # 加载数据
        self.images, self.labels = self._load_data()
        
        print(f"加载 {dataset_type} 数据集 - {split} 分割")
        print(f"样本数量: {len(self.images)}")
        print(f"类别数量: {self.num_classes}")
        print(f"类别名称: {self.class_names}")
    
    def _extract_dataset(self):
        """解压数据集压缩包"""
        dataset_path = os.path.join(self.data_dir, f"{self.dataset_type}")
        zip_path = os.path.join(self.data_dir, f"{self.dataset_type}.zip")
        
        # 如果数据集目录不存在但压缩包存在，则解压
        if self.dataset_type in ['NEU-DET', 'FSSD-12'] and (not os.path.exists(dataset_path) and os.path.exists(zip_path)):
            print(f"正在解压 {self.dataset_type}.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print(f"解压完成: {dataset_path}")
    
    def _load_data(self):
        """加载图像和标签数据"""
        dataset_path = os.path.join(self.data_dir, self.dataset_type)
        
        if self.dataset_type in ['NEU-DET', 'FSSD-12']:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        images = []
        labels = []
        
        if self.dataset_type == 'NEU-DET':
            # NEU-DET数据集结构
            for class_idx, folder_name in enumerate(self.folder_names):
                class_dir = os.path.join(dataset_path, folder_name)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_dir, img_name)
                            images.append(img_path)
                            labels.append(class_idx)
        
        elif self.dataset_type == 'FSSD-12':
            # FSSD-12数据集结构
            for class_idx, folder_name in enumerate(self.folder_names):
                class_dir = os.path.join(dataset_path, folder_name)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_dir, img_name)
                            images.append(img_path)
                            labels.append(class_idx)
        elif self.dataset_type == 'CIFAR-100':
            # 合并CIFAR-100训练与测试集，统一后续分割流程
            cifar_train = tv_datasets.CIFAR100(root=self.data_dir, train=True, download=True)
            cifar_test = tv_datasets.CIFAR100(root=self.data_dir, train=False, download=True)
            # torchvision 返回的是 (PIL.Image, label)
            for img, label in cifar_train:
                images.append(img.convert('RGB'))
                labels.append(int(label))
            for img, label in cifar_test:
                images.append(img.convert('RGB'))
                labels.append(int(label))
        
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_or_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图像
        if isinstance(img_or_path, str):
            try:
                image = Image.open(img_or_path).convert('RGB')
            except Exception as e:
                print(f"加载图像失败: {img_or_path}, 错误: {e}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            # 已经是PIL.Image对象（如CIFAR-100）
            image = img_or_path
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """获取类别分布"""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        distribution = {}
        for label, count in zip(unique_labels, counts):
            class_name = self.class_names[label]
            distribution[class_name] = count
        return distribution

def get_data_loaders(data_dir, dataset_type='NEU-DET', batch_size=32, 
                    train_split=0.7, val_split=0.15, test_split=0.15,
                    num_workers=4, transform_train=None, transform_val=None, data_ratio=1.0):
    """
    获取数据加载器
    
    Args:
        data_dir (str): 数据根目录
        dataset_type (str): 数据集类型
        batch_size (int): 批次大小
        train_split (float): 训练集比例
        val_split (float): 验证集比例
        test_split (float): 测试集比例
        num_workers (int): 数据加载器工作进程数
        transform_train: 训练集变换
        transform_val: 验证集变换
        data_ratio (float): 数据使用比例，1.0表示使用全部数据，0.3表示使用30%数据
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    # 创建完整数据集
    full_dataset = SteelDefectDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        split='full',
        transform=None
    )
    
    # 获取所有图像路径和标签
    all_images = full_dataset.images
    all_labels = full_dataset.labels
    
    # 如果data_ratio < 1.0，先进行数据采样
    if data_ratio < 1.0:
        print(f"使用 {data_ratio*100:.1f}% 的数据进行训练...")
        # 分层采样
        _, sampled_images, _, sampled_labels = train_test_split(
            all_images, all_labels,
            test_size=data_ratio,
            random_state=42,
            stratify=all_labels
        )
        all_images = sampled_images
        all_labels = sampled_labels
        print(f"采样后数据量: {len(all_images)} 样本")
    
    # 分层分割数据集
    if val_split == 0.0:
        # 如果没有验证集，直接分割训练集和测试集
        train_images, test_images, train_labels, test_labels = train_test_split(
            all_images, all_labels,
            test_size=test_split,
            random_state=42,
            stratify=all_labels
        )
        val_images, val_labels = [], []
    else:
        # 有验证集的情况
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            all_images, all_labels, 
            test_size=(val_split + test_split), 
            random_state=42, 
            stratify=all_labels
        )
        
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels,
            test_size=test_split/(val_split + test_split),
            random_state=42,
            stratify=temp_labels
        )
    
    # 创建数据集对象
    train_dataset = SteelDatasetSplit(
        train_images, train_labels, transform_train
    )
    
    if len(val_images) > 0:
        val_dataset = SteelDatasetSplit(
            val_images, val_labels, transform_val
        )
    else:
        val_dataset = None
        
    test_dataset = SteelDatasetSplit(
        test_images, test_labels, transform_val
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    else:
        val_loader = None
        
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"数据集分割完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    if val_dataset is not None:
        print(f"  验证集: {len(val_dataset)} 样本")
    else:
        print(f"  验证集: 0 样本 (无验证集)")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader, full_dataset.class_names

class SteelDatasetSplit(Dataset):
    """数据集分割类"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_or_path = self.images[idx]
        label = self.labels[idx]

        if isinstance(img_or_path, str):
            try:
                image = Image.open(img_or_path).convert('RGB')
            except Exception as e:
                print(f"加载图像失败: {img_or_path}, 错误: {e}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            # 已经是PIL.Image（例如CIFAR-100）
            image = img_or_path

        if self.transform:
            image = self.transform(image)

        return image, label

def analyze_dataset(data_dir, dataset_type='NEU-DET'):
    """
    分析数据集统计信息
    
    Args:
        data_dir (str): 数据根目录
        dataset_type (str): 数据集类型
    """
    print(f"分析 {dataset_type} 数据集...")
    
    # 创建数据集（对于CIFAR-100，会自动下载/加载）
    dataset = SteelDefectDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        split='full',
        transform=None
    )
    
    # 获取类别分布
    distribution = dataset.get_class_distribution()
    
    print(f"\n数据集统计信息:")
    print(f"总样本数: {len(dataset)}")
    print(f"类别数: {dataset.num_classes}")
    print(f"类别分布:")
    for class_name, count in distribution.items():
        percentage = count / len(dataset) * 100
        print(f"  {class_name}: {count} 样本 ({percentage:.1f}%)")
    
    return distribution
