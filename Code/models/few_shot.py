import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class PrototypicalNetworks(nn.Module):
    """
    Prototypical Networks for Few-shot Learning
    基于原型的少样本学习方法
    """
    
    def __init__(self, backbone, num_classes=6, embedding_dim=64):
        """
        初始化原型网络
        
        Args:
            backbone: 特征提取网络
            num_classes: 类别数量
            embedding_dim: 嵌入维度
        """
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # 修改backbone的最后一层
        if hasattr(backbone, 'fc'):
            # ResNet类型
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier'):
            # MobileNet类型
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
        else:
            # 默认情况
            in_features = 512
            
        # 嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return embeddings
    
    def compute_prototypes(self, support_set, support_labels):
        """
        计算原型
        
        Args:
            support_set: 支持集 (N, C, H, W)
            support_labels: 支持集标签 (N,)
        
        Returns:
            prototypes: 每个类别的原型 (num_classes, embedding_dim)
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(support_set)
            
        prototypes = []
        for class_id in range(self.num_classes):
            class_mask = (support_labels == class_id)
            if class_mask.sum() > 0:
                class_embeddings = embeddings[class_mask]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)
            else:
                # 如果没有该类别的样本，使用零向量
                prototypes.append(torch.zeros(self.embedding_dim, device=embeddings.device))
        
        prototypes = torch.stack(prototypes)
        return prototypes
    
    def predict(self, query_set, support_set, support_labels):
        """
        预测查询集标签
        
        Args:
            query_set: 查询集 (M, C, H, W)
            support_set: 支持集 (N, C, H, W)
            support_labels: 支持集标签 (N,)
        
        Returns:
            predictions: 预测标签 (M,)
            distances: 到每个原型的距离 (M, num_classes)
        """
        # 计算原型
        prototypes = self.compute_prototypes(support_set, support_labels)
        
        # 计算查询集嵌入
        query_embeddings = self.forward(query_set)
        
        # 计算欧几里得距离
        distances = torch.cdist(query_embeddings, prototypes)
        
        # 预测标签（距离最小的类别）
        predictions = torch.argmin(distances, dim=1)
        
        return predictions, distances
    
    def prototypical_loss(self, support_set, support_labels, query_set, query_labels):
        """
        计算原型网络损失
        
        Args:
            support_set: 支持集
            support_labels: 支持集标签
            query_set: 查询集
            query_labels: 查询集标签
        
        Returns:
            loss: 原型网络损失
            accuracy: 准确率
        """
        # 计算原型
        prototypes = self.compute_prototypes(support_set, support_labels)
        
        # 计算查询集嵌入
        query_embeddings = self.forward(query_set)
        
        # 计算距离
        distances = torch.cdist(query_embeddings, prototypes)
        
        # 计算负对数似然损失
        log_probabilities = F.log_softmax(-distances, dim=1)
        loss = F.nll_loss(log_probabilities, query_labels)
        
        # 计算准确率
        predictions = torch.argmin(distances, dim=1)
        accuracy = (predictions == query_labels).float().mean()
        
        return loss, accuracy

class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML)
    模型无关的元学习方法
    """
    
    def __init__(self, backbone, num_classes=6, inner_lr=0.01, meta_lr=0.001):
        """
        初始化MAML
        
        Args:
            backbone: 基础网络
            num_classes: 类别数量
            inner_lr: 内循环学习率
            meta_lr: 元学习率
        """
        super(MAML, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        
        # 分类头
        if hasattr(backbone, 'fc'):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier'):
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
        else:
            in_features = 512
            
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def inner_loop(self, support_set, support_labels, num_steps=5):
        """
        内循环：在支持集上快速适应
        
        Args:
            support_set: 支持集
            support_labels: 支持集标签
            num_steps: 内循环步数
        
        Returns:
            adapted_params: 适应后的参数
        """
        # 克隆当前参数
        adapted_params = {}
        for name, param in self.named_parameters():
            adapted_params[name] = param.clone()
        
        # 内循环梯度更新
        for step in range(num_steps):
            # 前向传播
            logits = self.forward_with_params(support_set, adapted_params)
            loss = F.cross_entropy(logits, support_labels)
            
            # 计算梯度
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True, allow_unused=True)
            
            # 更新参数
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
                else:
                    # 如果梯度为None，保持原参数不变
                    adapted_params[name] = param
        
        return adapted_params
    
    def forward_with_params(self, x, params):
        """使用指定参数进行前向传播"""
        # 使用指定参数计算特征
        features = self.backbone(x)
        
        # 使用指定的分类器参数
        if 'classifier.weight' in params and 'classifier.bias' in params:
            weight = params['classifier.weight']
            bias = params['classifier.bias']
            logits = F.linear(features, weight, bias)
        else:
            # 如果没有分类器参数，使用默认分类器
            logits = self.classifier(features)
        
        return logits
    
    def meta_loss(self, support_sets, support_labels, query_sets, query_labels, num_inner_steps=5):
        """
        计算元学习损失
        
        Args:
            support_sets: 支持集列表
            support_labels: 支持集标签列表
            query_sets: 查询集列表
            query_labels: 查询集标签列表
            num_inner_steps: 内循环步数
        
        Returns:
            meta_loss: 元学习损失
            accuracies: 每个任务的准确率
        """
        meta_losses = []
        accuracies = []
        
        for support_set, support_label, query_set, query_label in zip(
            support_sets, support_labels, query_sets, query_labels
        ):
            # 内循环适应
            adapted_params = self.inner_loop(support_set, support_label, num_inner_steps)
            
            # 在查询集上评估
            query_logits = self.forward_with_params(query_set, adapted_params)
            query_loss = F.cross_entropy(query_logits, query_label)
            
            # 计算准确率
            predictions = torch.argmax(query_logits, dim=1)
            accuracy = (predictions == query_label).float().mean()
            
            meta_losses.append(query_loss)
            accuracies.append(accuracy)
        
        # 平均元损失
        total_meta_loss = torch.stack(meta_losses).mean()
        avg_accuracy = torch.stack(accuracies).mean()
        
        return total_meta_loss, avg_accuracy

class FewShotTrainer:
    """
    小样本学习训练器
    """
    
    def __init__(self, model, device, num_ways=6, num_shots=5, num_queries=15):
        """
        初始化训练器
        
        Args:
            model: 小样本学习模型
            device: 设备
            num_ways: 每个episode的类别数
            num_shots: 每个类别的支持样本数
            num_queries: 每个类别的查询样本数
        """
        self.model = model
        self.device = device
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        
    def create_episode(self, data_loader, num_ways=None, num_shots=None, num_queries=None):
        """
        创建一个episode
        
        Args:
            data_loader: 数据加载器
            num_ways: 类别数
            num_shots: 支持样本数
            num_queries: 查询样本数
        
        Returns:
            support_set, support_labels, query_set, query_labels
        """
        if num_ways is None:
            num_ways = self.num_ways
        if num_shots is None:
            num_shots = self.num_shots
        if num_queries is None:
            num_queries = self.num_queries
            
        # 随机选择类别
        all_classes = list(range(num_ways))
        selected_classes = np.random.choice(all_classes, num_ways, replace=False)
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        # 为每个类别收集数据
        for class_idx, class_id in enumerate(selected_classes):
            class_data = []
            class_labels = []
            
            # 从数据加载器中收集该类别的数据
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                class_mask = (batch_labels == class_id)
                if class_mask.sum() > 0:
                    class_samples = batch_data[class_mask]
                    class_labels_batch = batch_labels[class_mask]
                    
                    class_data.append(class_samples)
                    class_labels.append(class_labels_batch)
                
                if len(torch.cat(class_data)) >= num_shots + num_queries:
                    break
            
            if len(class_data) == 0:
                continue
                
            class_data = torch.cat(class_data)
            class_labels = torch.cat(class_labels)
            
            # 确保有足够的样本
            if len(class_data) < num_shots + num_queries:
                # 如果样本不足，重复采样
                repeat_times = (num_shots + num_queries) // len(class_data) + 1
                class_data = class_data.repeat(repeat_times, 1, 1, 1)
                class_labels = class_labels.repeat(repeat_times)
            
            # 随机选择支持集和查询集
            indices = torch.randperm(len(class_data))
            support_indices = indices[:num_shots]
            query_indices = indices[num_shots:num_shots + num_queries]
            
            support_data.append(class_data[support_indices])
            support_labels.append(torch.full((num_shots,), class_idx, device=self.device))
            query_data.append(class_data[query_indices])
            query_labels.append(torch.full((num_queries,), class_idx, device=self.device))
        
        if len(support_data) == 0:
            return None, None, None, None
            
        support_set = torch.cat(support_data)
        support_labels = torch.cat(support_labels)
        query_set = torch.cat(query_data)
        query_labels = torch.cat(query_labels)
        
        return support_set, support_labels, query_set, query_labels
    
    def train_episode(self, support_set, support_labels, query_set, query_labels):
        """
        训练一个episode
        
        Args:
            support_set: 支持集
            support_labels: 支持集标签
            query_set: 查询集
            query_labels: 查询集标签
        
        Returns:
            loss: 损失
            accuracy: 准确率
        """
        if isinstance(self.model, PrototypicalNetworks):
            loss, accuracy = self.model.prototypical_loss(
                support_set, support_labels, query_set, query_labels
            )
        elif isinstance(self.model, MAML):
            # MAML需要多个任务，这里简化为单个任务
            loss, accuracy = self.model.meta_loss(
                [support_set], [support_labels], [query_set], [query_labels]
            )
        else:
            raise ValueError("不支持的模型类型")
        
        return loss, accuracy
    
    def evaluate(self, data_loader, num_episodes=100):
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            num_episodes: 评估episode数
        
        Returns:
            accuracy: 平均准确率
        """
        self.model.eval()
        accuracies = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                support_set, support_labels, query_set, query_labels = self.create_episode(data_loader)
                
                if support_set is None:
                    continue
                
                if isinstance(self.model, PrototypicalNetworks):
                    predictions, _ = self.model.predict(support_set, support_labels, query_set)
                    accuracy = (predictions == query_labels).float().mean()
                elif isinstance(self.model, MAML):
                    adapted_params = self.model.inner_loop(support_set, support_labels)
                    query_logits = self.model.forward_with_params(query_set, adapted_params)
                    predictions = torch.argmax(query_logits, dim=1)
                    accuracy = (predictions == query_labels).float().mean()
                
                accuracies.append(accuracy.item())
        
        return np.mean(accuracies) if accuracies else 0.0
