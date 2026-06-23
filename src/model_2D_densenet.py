import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer2D(nn.Module):
    """
    2D DenseNet 核心密集层 (BottleNeck 结构)
    包含: 1x1 卷积压缩通道 -> 3x3 卷积提取特征
    """
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer2D, self).__init__()
        # DenseNet 经典设计：内部瓶颈通道数固定为 4 * growth_rate
        inter_channels = 4 * growth_rate
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # x 为包含之前所有特征图拼接后的输入
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # 在通道维度拼接原始输入与新提取的特征 (Dense Connection)
        return torch.cat([x, out], 1)


class DenseBlock2D(nn.Module):
    """
    2D DenseNet 密集块 (Dense Block)
    由多个 DenseLayer 串联组成，每个层的输入都是前面所有层输出的叠加
    """
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock2D, self).__init__()
        layers = []
        for i in range(num_layers):
            # 每经过一层，输入通道数增加 growth_rate
            layers.append(DenseLayer2D(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Transition2D(nn.Module):
    """
    2D 过渡层 (Transition Layer)
    用于在两个 Dense Block 之间控制通道数并减半空间特征图的分辨率
    """
    def __init__(self, in_channels, out_channels):
        super(Transition2D, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet2D121(nn.Module):
    """
    针对 2D 肺结节分类量身定制的轻量级 DenseNet-121 网络
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=2):
        super(DenseNet2D121, self).__init__()
        self.growth_rate = growth_rate
        
        # 1. 初始前置 2D 卷积层 (输入通道 C=1，输出 64 维特征)
        # 严格对齐医学图像单通道输入规范
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 跟踪当前通道数
        num_features = 64
        
        # 2. 构建 Dense Blocks 与 Transitions
        self.features = nn.Sequential()
        
        # 逐层构建网络
        for i, num_layers in enumerate(block_config):
            # 添加 Dense Block
            block = DenseBlock2D(num_layers=num_layers, in_channels=num_features, growth_rate=growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # 如果不是最后一个 Dense Block，在后面挂载 Transition 过渡层进行下采样
            if i != len(block_config) - 1:
                # 压缩通道数（压缩率 0.5）
                trans = Transition2D(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
                
        # 3. 全局平均池化与分类头
        self.final_bn = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # 初始特征提取
        out = F.relu(self.bn1(self.conv1(x)))
        # 密集特征传导
        out = self.features(out)
        out = F.relu(self.final_bn(out))
        # 池化与展平
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # 分类输出
        out = self.fc(out)
        return out

# ==========================================
# 本地模型前向传播测试 (Local Test Entry)
# ==========================================
if __name__ == "__main__":
    # 模拟一个 Batch 的 2D 横截面切片数据：Batch Size=4，单通道，大小为 32x32
    mock_input = torch.randn(4, 1, 32, 32)
    print("=== 2D DenseNet-121 架构测试 ===")
    print(f"输入伪张量形状 (Input Shape): {mock_input.shape}")
    
    # 初始化模型
    model = DenseNet2D121(num_classes=2)
    
    # 前向传播测试
    try:
        mock_output = model(mock_input)
        print(f"输出分类张量形状 (Output Shape): {mock_output.shape}") # 期望输出: torch.Size([4, 2])
        print("✅ 2D-DenseNet121 架构搭建成功，前向传播测试通过")
    except Exception as e:
        print(f"❌ 架构测试失败: {e}")