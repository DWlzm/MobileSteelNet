import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import time
import json
import random
import numpy as np
from datetime import datetime

from models import ResNet18, ResNet34, ResNet50, MobileNetV1, MobileNetV2, MobileNetV3, ShuffleNetV1, ShuffleNetV2
from models.mobilenet_enhanced import create_enhanced_mobilenet
from dataset import get_data_loaders, get_transforms, analyze_dataset
from utils import MetricsCalculator, plot_training_curves, create_visualization_report

class SteelDefectTrainer:
    """
    钢铁表面缺陷分类训练器
    """
    
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config (dict): 训练配置
        """
        self.config = config
        
        # 设置随机种子
        self._set_random_seed(config.get('random_seed', 42))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建保存目录
        self.save_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 初始化优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 初始化损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 初始化指标计算器
        self.metrics_calculator = MetricsCalculator(
            class_names=config.get('class_names', None)
        )
        
        # 训练历史
        self.train_history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'learning_rates': []
        }
        
        # 最佳模型指标
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
    
    def _set_random_seed(self, seed):
        """
        设置随机种子以确保结果可复现
        
        Args:
            seed (int): 随机种子
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 设置PyTorch的确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"设置随机种子: {seed}")
        
    def _create_model(self):
        """创建模型"""
        model_name = self.config['model_name']
        num_classes = self.config.get('num_classes', 6)
        ema_type = self.config.get('ema_type', 'original')
        width_mult = self.config.get('width_mult', 1.0)
        
        # 检查是否是增强版MobileNet
        if model_name.startswith('Enhanced'):
            base_model = model_name.replace('Enhanced', '')
            model = create_enhanced_mobilenet(base_model, ema_type, num_classes, width_mult)
            print(f"创建增强版模型: {model_name} (EMA类型: {ema_type})")
        else:
            model_dict = {
                'ResNet18': ResNet18,
                'ResNet34': ResNet34,
                'ResNet50': ResNet50,
                'MobileNetV1': MobileNetV1,
                'MobileNetV2': MobileNetV2,
                'MobileNetV3': MobileNetV3,
                'ShuffleNetV1': ShuffleNetV1,
                'ShuffleNetV2': ShuffleNetV2
            }
            
            if model_name not in model_dict:
                raise ValueError(f"不支持的模型: {model_name}")
            
            model = model_dict[model_name](num_classes=num_classes)
            print(f"创建模型: {model_name}")
        
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model
    
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_name = self.config.get('optimizer', 'Adam')
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            momentum = self.config.get('momentum', 0.9)
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, 
                                momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        print(f"使用优化器: {optimizer_name}, 学习率: {learning_rate}")
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_name = self.config.get('scheduler', 'StepLR')
        
        if scheduler_name == 'StepLR':
            step_size = self.config.get('step_size', 30)
            gamma = self.config.get('gamma', 0.1)
            scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'CosineAnnealingLR':
            T_max = self.config.get('T_max', 100)
            scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10)
        else:
            scheduler = None
        
        if scheduler:
            print(f"使用学习率调度器: {scheduler_name}")
        
        return scheduler
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 打印进度
            if batch_idx % 100 == 0:
                print(f'批次 [{batch_idx}/{len(train_loader)}], 损失: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, test_loader=None):
        """训练模型"""
        print("开始训练...")
        print(f"训练轮次: {self.config['epochs']}")
        print(f"批次大小: {self.config['batch_size']}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f"\n轮次 {epoch+1}/{self.config['epochs']}")
            print("-" * 30)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.train_history['train_losses'].append(train_loss)
            self.train_history['train_accuracies'].append(train_acc)
            self.train_history['val_losses'].append(val_loss)
            self.train_history['val_accuracies'].append(val_acc)
            self.train_history['learning_rates'].append(current_lr)
            
            # 打印结果
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                print(f"保存最佳模型 (验证准确率: {val_acc:.2f}%)")
            
            # 定期保存检查点
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch)
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"\n训练完成!")
        print(f"总训练时间: {training_time:.2f}秒")
        print(f"最佳验证准确率: {self.best_val_accuracy:.2f}% (轮次 {self.best_epoch+1})")
        
        # 在测试集上评估
        if test_loader:
            print("\n在测试集上评估...")
            self.evaluate(test_loader, save_results=True)
    
    def evaluate(self, test_loader, save_results=False):
        """评估模型"""
        print("评估模型...")
        
        self.model.eval()
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                # 更新指标
                self.metrics_calculator.update(predicted, target, output)
        
        # 打印指标
        self.metrics_calculator.print_metrics()
        
        # 保存结果
        if save_results:
            results_dir = os.path.join(self.save_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # 保存指标
            self.metrics_calculator.save_metrics(
                os.path.join(results_dir, 'metrics.txt')
            )
            
            # 创建可视化报告
            create_visualization_report(
                self.metrics_calculator,
                self.train_history,
                results_dir
            )
            
            # 保存训练历史
            with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
                json.dump(self.train_history, f, indent=2)
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'train_history': self.train_history,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存检查点
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        self.train_history = checkpoint.get('train_history', self.train_history)
        
        print(f"加载检查点: {checkpoint_path}")
        print(f"轮次: {checkpoint['epoch']+1}")
        print(f"最佳验证准确率: {self.best_val_accuracy:.2f}%")

def main():
    """主函数"""
    # 训练配置
    config = {
        # 数据配置
        'data_dir': '/root/SteelClassification/Data',
        'dataset_type': 'NEU-DET',  # 可选: 'NEU-DET', 'FSSD-12', 'CIFAR-100'
        'batch_size': 16,
        'num_workers': 8,
        'image_size': 224,

        # 模型配置
        'model_name': 'MobileNetV1',
        'num_classes': 6,  # NEU-DET 类别数
        'ema_type': 'enhanced',
        'width_mult': 1.0,

        # 训练配置
        'epochs': 100,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'weight_decay': 1e-4,
        'data_ratio': 1.0,
        'train_split': 0.05,
        'val_split': 0.05,
        'test_split': 0.9,

        # 保存配置
        'save_dir': 'checkpoints',
        'save_interval': 10,

        # 随机种子设置
        'random_seed': 42,

        # 类别名称（将由数据加载器覆盖）
        'class_names': []
    }
    
    print("钢铁表面缺陷分类训练")
    print("=" * 50)
    print(f"数据集: {config['dataset_type']}")
    print(f"模型: {config['model_name']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"训练轮次: {config['epochs']}")
    print("=" * 50)
    
    # 分析数据集
    print("分析数据集...")
    analyze_dataset(config['data_dir'], config['dataset_type'])
    
    # 获取数据加载器
    print("\n加载数据...")
    train_transform, val_transform = get_transforms(
        image_size=config['image_size'],
        augmentation=True
    )
    
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir=config['data_dir'],
        dataset_type=config['dataset_type'],
        batch_size=config['batch_size'],
        train_split=config.get('train_split', 0.7),
        val_split=config.get('val_split', 0.15),
        test_split=config.get('test_split', 0.15),
        num_workers=config['num_workers'],
        transform_train=train_transform,
        transform_val=val_transform,
        data_ratio=config.get('data_ratio', 1.0)
    )
    
    # 更新类别名称
    config['class_names'] = class_names
    config['num_classes'] = len(class_names)
    
    # 创建训练器
    trainer = SteelDefectTrainer(config)
    
    # 开始训练（如果没有验证集，使用测试集作为验证集）
    if val_loader is None or len(val_loader.dataset) == 0:
        print("没有验证集，使用测试集进行验证...")
        trainer.train(train_loader, test_loader, test_loader)
    else:
        trainer.train(train_loader, val_loader, test_loader)
    
    print("\n训练完成!")

if __name__ == '__main__':
    main()
