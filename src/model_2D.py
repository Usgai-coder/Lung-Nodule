import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock2D(nn.Module):
    """
    2D ResNet 核心残差块 (2D Residual Block)
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2D, self).__init__()
        # 第一层 2D 卷积
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 第二层 2D 卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 捷径分支 (Shortcut / Skip Connection)：当输入输出维度不一致时，用 1x1 卷积对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接：将输入直接与卷积输出相加
        out = F.relu(out)
        return out


class ResNet2D18(nn.Module):
    """
    针对 2D 肺结节分类量身定制的 2D-ResNet18 网络
    """
    def __init__(self, num_classes=2): # 2分类：良性 vs 恶性
        super(ResNet2D18, self).__init__()
        self.in_planes = 64

        # 1. 初始前置 2D 卷积层 (输入通道 C=1，输出 64 维特征)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 2. 四组残差层 (随着层数加深，分辨率减半，通道数翻倍)
        self.layer1 = self._make_layer(BasicBlock2D, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock2D, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock2D, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock2D, 512, num_blocks=2, stride=2)
        
        # 3. 二维自适应平均池化层 + 全连接分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock2D.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 提取特征
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 池化与展平
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # 输出分类对数概率 (Logits)
        out = self.fc(out)
        return out

# ==========================================
# 本地模型前向传播测试 (Local Test Entry)
# ==========================================
if __name__ == "__main__":
    # 模拟一个 Batch 的肺结节数据：假设 Batch Size=4，单通道，大小为 32x32 (丢弃了深度维度的2D最大截面)
    mock_input = torch.randn(4, 1, 32, 32)
    print(f"输入伪张量形状 (Input Shape): {mock_input.shape}")
    
    # 初始化模型
    model = ResNet2D18(num_classes=2)
    
    # 前向传播测试
    mock_output = model(mock_input)
    print(f"输出分类张量形状 (Output Shape): {mock_output.shape}") # 期望输出: torch.Size([4, 2])
    print("✅ 2D-ResNet18 架构搭建成功，前向传播测试通过")