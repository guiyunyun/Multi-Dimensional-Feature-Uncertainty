"""
Simple Classifier Head for Active Learning
- 轻量级线性分类头
- 接收DINOv3特征，输出类别logits
- 只有分类头可训练，骨干网络冻结
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_config():
    """获取 Config 对象"""
    try:
        from .config import Config
    except ImportError:
        from config import Config
    return Config


class SimpleClassifier(nn.Module):
    """
    简单的线性分类头
    
    结构:
    DINOv3特征 [B, 768] → Linear → [B, num_classes]
    """
    
    def __init__(
        self,
        input_dim: int = None,
        num_classes: int = None,
        dropout: float = None
    ):
        """
        Args:
            input_dim: 输入特征维度（DINOv3 ViT-B: 768, ViT-L: 1024）
            num_classes: 类别数
            dropout: Dropout比率（0表示不使用）
        """
        super().__init__()
        
        Config = _get_config()
        
        # 从 Config 读取默认值
        if input_dim is None:
            input_dim = getattr(Config, 'feature_dim', 768)
        if num_classes is None:
            num_classes = getattr(Config, 'num_classes', 100)
        if dropout is None:
            dropout = getattr(Config, 'classifier_dropout', 0.1)
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 简单的线性层
        self.fc = nn.Linear(input_dim, num_classes)
        
        # 可选的Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: [B, input_dim] DINOv3特征（CLS token）
        
        Returns:
            logits: [B, num_classes] 类别分数
        """
        features = self.dropout(features)
        logits = self.fc(features)
        return logits


class MLPClassifier(nn.Module):
    """
    多层感知机分类头（可选，更强的拟合能力）
    
    结构:
    DINOv3特征 [B, 768] → MLP → [B, num_classes]
    """
    
    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = None,
        num_classes: int = None,
        dropout: float = None
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            dropout: Dropout比率
        """
        super().__init__()
        
        Config = _get_config()
        
        # 从 Config 读取默认值
        if input_dim is None:
            input_dim = getattr(Config, 'feature_dim', 768)
        if hidden_dim is None:
            hidden_dim = getattr(Config, 'classifier_hidden_dim', 512)
        if num_classes is None:
            num_classes = getattr(Config, 'num_classes', 100)
        if dropout is None:
            dropout = getattr(Config, 'classifier_dropout', 0.1)
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 两层MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: [B, input_dim] DINOv3特征
        
        Returns:
            logits: [B, num_classes] 类别分数
        """
        return self.mlp(features)


def train_epoch(
    classifier: nn.Module,
    feature_extractor: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda'
) -> float:
    """
    训练一个epoch
    
    Args:
        classifier: 分类头（可训练）
        feature_extractor: 特征提取器（冻结）
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
    
    Returns:
        avg_loss: 平均损失
    """
    classifier.train()
    feature_extractor.eval()  # 确保冻结
    
    total_loss = 0.0
    num_batches = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 提取特征（不需要梯度）
        with torch.no_grad():
            features = feature_extractor.get_global_features(images)  # [B, 768]
        
        # 分类（需要梯度）
        logits = classifier(features)  # [B, num_classes]
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(
    classifier: nn.Module,
    feature_extractor: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    return_per_class: bool = False,
    num_classes: int = None
) -> dict:
    """
    评估模型性能
    
    Args:
        classifier: 分类头
        feature_extractor: 特征提取器
        dataloader: 数据加载器
        device: 设备
        return_per_class: 是否返回每个类别的准确率
        num_classes: 类别数（return_per_class=True时需要）
    
    Returns:
        metrics: {
            'accuracy': 准确率,
            'loss': 平均损失,
            'per_class_accuracy': [100个类别各自的准确率] (可选)
        }
    """
    classifier.eval()
    feature_extractor.eval()
    
    # 从 Config 读取 num_classes
    if num_classes is None:
        Config = _get_config()
        num_classes = getattr(Config, 'num_classes', 100)
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    num_batches = 0
    
    # 用于计算 per-class accuracy
    if return_per_class:
        class_correct = torch.zeros(num_classes)  # 每个类正确的数量
        class_total = torch.zeros(num_classes)    # 每个类总的数量
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 提取特征
        features = feature_extractor.get_global_features(images)
        
        # 分类
        logits = classifier(features)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        
        # 计算准确率
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        num_batches += 1
        
        # 计算 per-class accuracy
        if return_per_class:
            for c in range(num_classes):
                mask = (labels == c)
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == labels) & mask).sum().item()
    
    accuracy = total_correct / total_samples * 100
    avg_loss = total_loss / num_batches
    
    result = {
        'accuracy': accuracy,
        'loss': avg_loss
    }
    
    # 计算每个类别的准确率
    if return_per_class:
        per_class_acc = []
        for c in range(num_classes):
            if class_total[c] > 0:
                acc = (class_correct[c] / class_total[c] * 100).item()
            else:
                acc = 0.0  # 该类没有样本
            per_class_acc.append(round(acc, 2))
        result['per_class_accuracy'] = per_class_acc
    
    return result


def test_classifier():
    """测试分类头"""
    Config = _get_config()
    
    print("=" * 60)
    print("测试分类头")
    print("=" * 60)
    print()
    
    print(f"配置来源: config.py")
    print(f"  feature_dim: {Config.feature_dim}")
    print(f"  num_classes: {Config.num_classes}")
    print(f"  classifier_dropout: {Config.classifier_dropout}")
    print(f"  classifier_hidden_dim: {Config.classifier_hidden_dim}")
    print()
    
    # 测试SimpleClassifier（参数从 Config 读取）
    print("1. 测试SimpleClassifier:")
    classifier = SimpleClassifier()
    print(f"✓ SimpleClassifier已创建")
    print(f"  参数量: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # 测试前向传播
    batch_size = 4
    features = torch.randn(batch_size, Config.feature_dim)
    logits = classifier(features)
    print(f"  输入shape: {features.shape}")
    print(f"  输出shape: {logits.shape}")
    print(f"  输出范围: [{logits.min():.2f}, {logits.max():.2f}]")
    print()
    
    # 测试MLPClassifier（参数从 Config 读取）
    print("2. 测试MLPClassifier:")
    mlp_classifier = MLPClassifier()
    print(f"✓ MLPClassifier已创建")
    print(f"  参数量: {sum(p.numel() for p in mlp_classifier.parameters()):,}")
    
    logits = mlp_classifier(features)
    print(f"  输出shape: {logits.shape}")
    print()
    
    # 测试可训练性
    print("3. 测试可训练性:")
    print(f"  SimpleClassifier可训练参数: {sum(p.numel() for p in classifier.parameters() if p.requires_grad):,}")
    print(f"  MLPClassifier可训练参数: {sum(p.numel() for p in mlp_classifier.parameters() if p.requires_grad):,}")
    print()
    
    # 测试损失计算
    print("4. 测试损失计算:")
    labels = torch.randint(0, Config.num_classes, (batch_size,))
    loss = F.cross_entropy(logits, labels)
    print(f"  交叉熵损失: {loss.item():.4f}")
    print()
    
    print("=" * 60)
    print("✓ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_classifier()

