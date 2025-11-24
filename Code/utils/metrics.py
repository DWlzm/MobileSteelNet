import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsCalculator:
    """
    计算和输出各种评估指标的类
    包括混淆矩阵、各类别准确率、总准确率等
    """
    
    def __init__(self, class_names=None):
        """
        初始化指标计算器
        
        Args:
            class_names (list): 类别名称列表，如果为None则使用数字索引
        """
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """重置所有累积的指标"""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, predictions, targets, probabilities=None):
        """
        更新预测结果
        
        Args:
            predictions (torch.Tensor or np.ndarray): 预测的类别
            targets (torch.Tensor or np.ndarray): 真实标签
            probabilities (torch.Tensor or np.ndarray, optional): 预测概率
        """
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        if probabilities is not None and torch.is_tensor(probabilities):
            probabilities = probabilities.cpu().numpy()
        
        self.all_predictions.extend(predictions)
        self.all_targets.extend(targets)
        if probabilities is not None:
            self.all_probabilities.extend(probabilities)
    
    def compute_accuracy(self):
        """
        计算总准确率
        
        Returns:
            float: 总准确率
        """
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        accuracy = np.mean(predictions == targets)
        return accuracy
    
    def compute_class_accuracy(self):
        """
        计算各类别准确率
        
        Returns:
            dict: 各类别准确率字典
        """
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        unique_classes = np.unique(targets)
        class_accuracies = {}
        
        for cls in unique_classes:
            mask = targets == cls
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == targets[mask])
                class_name = self.class_names[cls] if self.class_names else f"Class_{cls}"
                class_accuracies[class_name] = class_acc
            else:
                class_name = self.class_names[cls] if self.class_names else f"Class_{cls}"
                class_accuracies[class_name] = 0.0
        
        return class_accuracies
    
    def compute_confusion_matrix(self):
        """
        计算混淆矩阵
        
        Returns:
            np.ndarray: 混淆矩阵
        """
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # 获取所有类别
        all_classes = np.unique(np.concatenate([targets, predictions]))
        cm = confusion_matrix(targets, predictions, labels=all_classes)
        return cm
    
    def compute_precision_recall_f1(self):
        """
        计算精确率、召回率和F1分数
        
        Returns:
            dict: 包含各类别指标的字典
        """
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        report = classification_report(targets, predictions, 
                                     target_names=self.class_names,
                                     output_dict=True,
                                     zero_division=0)
        return report
    
    def print_metrics(self):
        """
        打印所有指标
        """
        print("=" * 50)
        print("模型评估指标")
        print("=" * 50)
        
        # 总准确率
        total_accuracy = self.compute_accuracy()
        print(f"总准确率: {total_accuracy:.4f} ({total_accuracy*100:.2f}%)")
        print()
        
        # 各类别准确率
        class_accuracies = self.compute_class_accuracy()
        print("各类别准确率:")
        for class_name, acc in class_accuracies.items():
            print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")
        print()
        
        # 混淆矩阵
        cm = self.compute_confusion_matrix()
        print("混淆矩阵:")
        print(cm)
        print()
        
        # 详细报告
        report = self.compute_precision_recall_f1()
        print("详细分类报告:")
        print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10}")
        print("-" * 60)
        
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                precision = metrics['precision']
                recall = metrics['recall']
                f1 = metrics['f1-score']
                support = metrics['support']
                print(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
        
        # 宏平均和微平均
        if 'macro avg' in report:
            macro_avg = report['macro avg']
            print(f"{'宏平均':<15} {macro_avg['precision']:<10.4f} {macro_avg['recall']:<10.4f} {macro_avg['f1-score']:<10.4f} {macro_avg['support']:<10}")
        
        if 'weighted avg' in report:
            weighted_avg = report['weighted avg']
            print(f"{'加权平均':<15} {weighted_avg['precision']:<10.4f} {weighted_avg['recall']:<10.4f} {weighted_avg['f1-score']:<10.4f} {weighted_avg['support']:<10}")
        
        print("=" * 50)
    
    def save_metrics(self, filepath):
        """
        保存指标到文件
        
        Args:
            filepath (str): 保存路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("模型评估指标\n")
            f.write("=" * 50 + "\n")
            
            # 总准确率
            total_accuracy = self.compute_accuracy()
            f.write(f"总准确率: {total_accuracy:.4f} ({total_accuracy*100:.2f}%)\n\n")
            
            # 各类别准确率
            class_accuracies = self.compute_class_accuracy()
            f.write("各类别准确率:\n")
            for class_name, acc in class_accuracies.items():
                f.write(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)\n")
            f.write("\n")
            
            # 混淆矩阵
            cm = self.compute_confusion_matrix()
            f.write("混淆矩阵:\n")
            f.write(str(cm))
            f.write("\n\n")
            
            # 详细报告
            report = self.compute_precision_recall_f1()
            f.write("详细分类报告:\n")
            f.write(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10}\n")
            f.write("-" * 60 + "\n")
            
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    precision = metrics['precision']
                    recall = metrics['recall']
                    f1 = metrics['f1-score']
                    support = metrics['support']
                    f.write(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}\n")
            
            if 'macro avg' in report:
                macro_avg = report['macro avg']
                f.write(f"{'宏平均':<15} {macro_avg['precision']:<10.4f} {macro_avg['recall']:<10.4f} {macro_avg['f1-score']:<10.4f} {macro_avg['support']:<10}\n")
            
            if 'weighted avg' in report:
                weighted_avg = report['weighted avg']
                f.write(f"{'加权平均':<15} {weighted_avg['precision']:<10.4f} {weighted_avg['recall']:<10.4f} {weighted_avg['f1-score']:<10.4f} {weighted_avg['support']:<10}\n")
    
    def get_summary(self):
        """
        获取指标摘要
        
        Returns:
            dict: 包含主要指标的字典
        """
        total_accuracy = self.compute_accuracy()
        class_accuracies = self.compute_class_accuracy()
        report = self.compute_precision_recall_f1()
        
        summary = {
            'total_accuracy': total_accuracy,
            'class_accuracies': class_accuracies,
            'confusion_matrix': self.compute_confusion_matrix(),
            'classification_report': report
        }
        
        return summary

