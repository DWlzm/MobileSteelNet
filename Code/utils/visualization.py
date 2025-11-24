import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, figsize=(10, 8)):
    """
    绘制混淆矩阵热力图
    
    Args:
        y_true (array-like): 真实标签
        y_pred (array-like): 预测标签
        class_names (list): 类别名称列表
        save_path (str): 保存路径，如果为None则显示图像
        figsize (tuple): 图像大小
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 设置英文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图像
    plt.figure(figsize=figsize)
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, 
                        save_path=None, figsize=(15, 5)):
    """
    绘制训练曲线
    
    Args:
        train_losses (list): 训练损失
        train_accuracies (list): 训练准确率
        val_losses (list): 验证损失
        val_accuracies (list): 验证准确率
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    # 设置英文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_class_distribution(labels, class_names=None, save_path=None, figsize=(10, 6)):
    """
    绘制类别分布图
    
    Args:
        labels (array-like): 标签数组
        class_names (list): 类别名称列表
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    # 设置英文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算类别分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [f'类别 {i}' for i in unique_labels]
    
    # 创建图像
    plt.figure(figsize=figsize)
    
    # 绘制柱状图
    bars = plt.bar(class_names, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Dataset Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_learning_rate_schedule(optimizer, epochs, save_path=None, figsize=(10, 6)):
    """
    绘制学习率调度图
    
    Args:
        optimizer: 优化器对象
        epochs (int): 总轮次
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    # 设置英文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取学习率调度器
    scheduler = optimizer.param_groups[0].get('lr_scheduler', None)
    
    if scheduler is None:
        print("未找到学习率调度器")
        return
    
    # 模拟学习率变化
    lr_history = []
    for epoch in range(epochs):
        lr_history.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    # 创建图像
    plt.figure(figsize=figsize)
    plt.plot(range(epochs), lr_history, 'b-', linewidth=2, marker='o', markersize=4)
    plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate schedule plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_visualization_report(metrics_calculator, train_history, save_dir):
    """
    创建完整的可视化报告
    
    Args:
        metrics_calculator: 指标计算器对象
        train_history (dict): 训练历史记录
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制混淆矩阵
    predictions = np.array(metrics_calculator.all_predictions)
    targets = np.array(metrics_calculator.all_targets)
    plot_confusion_matrix(targets, predictions, 
                         class_names=metrics_calculator.class_names,
                         save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    
    # 绘制训练曲线
    if 'train_losses' in train_history and 'val_losses' in train_history:
        plot_training_curves(
            train_history['train_losses'],
            train_history['train_accuracies'],
            train_history['val_losses'],
            train_history['val_accuracies'],
            save_path=os.path.join(save_dir, 'training_curves.png')
        )
    
    # 绘制类别分布
    plot_class_distribution(targets, 
                          class_names=metrics_calculator.class_names,
                          save_path=os.path.join(save_dir, 'class_distribution.png'))
    
    print(f"Visualization report saved to: {save_dir}")
