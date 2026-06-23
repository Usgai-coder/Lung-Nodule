import torch
import torch.nn as nn
import torch.nn.functional as F

class VNetConvBlock3D(nn.Module):
    """
    3D VNet 核心特征提取模块
    包含: 多个 3D 卷积层串联，带有残差短路连接，使用 PReLU 作为激活函数以适应医学影像
    """
    def __init__(self, channels, num_convs):
        super(VNetConvBlock3D, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv3d(channels, channels, kernel_size=5, padding=2, bias=False))
            layers.append(nn.BatchNorm3d(channels))
            layers.append(nn.PReLU(channels)) # VNet 标志性的 PReLU 激活
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        # 经典的局部残差相加设计
        return self.conv_block(x) + x


class VNet3DClassifier(nn.Module):
    """
    针对 3D 肺结节分类量身定制的 VNet-Classifier 网络。
    保留了 VNet 原作经典的下采样（Encoder）路径和基于 5x5x5 卷积核的大感受野提取机制，
    末端舍弃分割用的上采样 Decoder，挂载 3D 全局池化和分类器头。
    """
    def __init__(self, num_classes=2):
        super(VNet3DClassifier, self).__init__()
        
        # 1. 第一阶段 (输入 1x32x32x32 -> 输出 16x32x32x32)
        self.in_conv = nn.Conv3d(1, 16, kernel_size=5, padding=2, bias=False)
        self.in_bn = nn.BatchNorm3d(16)
        self.in_prelu = nn.PReLU(16)
        
        # 2. 第二阶段 (下采样 -> 32x16x16x16)
        self.down1 = nn.Conv3d(16, 32, kernel_size=2, stride=2, bias=False) # 用步长为2的 2x2x2 卷积代替Pooling
        self.down1_bn = nn.BatchNorm3d(32)
        self.down1_prelu = nn.PReLU(32)
        self.block1 = VNetConvBlock3D(32, num_convs=2)
        
        # 3. 第三阶段 (下采样 -> 64x8x8x8)
        self.down2 = nn.Conv3d(32, 64, kernel_size=2, stride=2, bias=False)
        self.down2_bn = nn.BatchNorm3d(64)
        self.down2_prelu = nn.PReLU(64)
        self.block2 = VNetConvBlock3D(64, num_convs=3)
        
        # 4. 第四阶段 (下采样 -> 128x4x4x4)
        self.down3 = nn.Conv3d(64, 128, kernel_size=2, stride=2, bias=False)
        self.down3_bn = nn.BatchNorm3d(128)
        self.down3_prelu = nn.PReLU(128)
        self.block3 = VNetConvBlock3D(128, num_convs=3)

        # 5. 第五阶段 (下采样 -> 256x2x2x2)
        self.down4 = nn.Conv3d(128, 256, kernel_size=2, stride=2, bias=False)
        self.down4_bn = nn.BatchNorm3d(256)
        self.down4_prelu = nn.PReLU(256)
        self.block4 = VNetConvBlock3D(256, num_convs=3)
        
        # 6. 三维自适应平均池化 + 分类输出层
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # 阶段 1
        out = self.in_prelu(self.in_bn(self.in_conv(x)))
        
        # 阶段 2 (下采样 + 特征提取)
        out = self.down1_prelu(self.down1_bn(self.down1(out)))
        out = self.block1(out)
        
        # 阶段 3
        out = self.down2_prelu(self.down2_bn(self.down2(out)))
        out = self.block2(out)
        
        # 阶段 4
        out = self.down3_prelu(self.down3_bn(self.down3(out)))
        out = self.block3(out)

        # 阶段 5
        out = self.down4_prelu(self.down4_bn(self.down4(out)))
        out = self.block4(out)
        
        # 池化与最终展平
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # 分类
        out = self.fc(out)
        return out

# ==========================================
# 本地模型前向传播测试 (Local Test Entry)
# ==========================================
if __name__ == "__main__":
    # 模拟一个 Batch 的 3D 体积数据：Batch Size=4，单通道，大小为 32x32x32
    mock_input = torch.randn(4, 1, 32, 32, 32)
    print("=== 3D VNet-Classifier 架构测试 ===")
    print(f"输入伪张量形状 (Input Shape): {mock_input.shape}")
    
    # 初始化模型
    model = VNet3DClassifier(num_classes=2)
    
    # 前向传播测试
    try:
        mock_output = model(mock_input)
        print(f"输出分类张量形状 (Output Shape): {mock_output.shape}") # 期望输出: torch.Size([4, 2])
        print("✅ 3D-VNet 架构搭建成功，前向传播测试通过")
    except Exception as e:
        print(f"❌ 架构测试失败: {e}")