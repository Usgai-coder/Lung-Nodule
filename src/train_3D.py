import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# 引入之前写好的组件
from dataset_3D import LungNodule3DDataset
from model_3D import ResNet3D18

# ==========================================
# 🧩 1. 3D 数据增强 (Data Augmentation)
# ==========================================
class Transform3D:
    """
    针对 3D 医疗影像的随机三维翻转增强
    """
    def __call__(self, tensor):
        # 50% 概率在 Depth (D轴) 翻转
        if torch.rand(1) > 0.5:
            tensor = torch.flip(tensor, dims=[1])
        # 50% 概率在 Height (H轴) 翻转
        if torch.rand(1) > 0.5:
            tensor = torch.flip(tensor, dims=[2])
        # 50% 概率在 Width (W轴) 翻转
        if torch.rand(1) > 0.5:
            tensor = torch.flip(tensor, dims=[3])
        return tensor

# ==========================================
# 🧠 2. PyTorch Lightning 核心模型封装
# ==========================================
class LitLungNoduleModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # 实例化我们的 3D-ResNet18
        self.model = ResNet3D18(num_classes=2)
        # 交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 计算当前 Batch 的准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # 记录日志 (prog_bar=True 会直接显示在终端进度条上)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    # ==========================================
    # 📉 学习率调度器配置 (LR Scheduler)
    # ==========================================
    def configure_optimizers(self):
        # 使用 AdamW 优化器
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        
        # 学习率衰减：如果验证集 val_loss 连续 5 个 epoch 没下降，学习率减半 (factor=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

# ==========================================
# 🚀 3. 自动化训练执行入口
# ==========================================
if __name__ == "__main__":
    # 路径配置
    MANIFEST_PATH = "./data/processed/final_dataset_manifest.csv"
    CUBE_DIR = "./data/processed/all_roi_cubes/"
    
    # 初始化数据集并注入 3D 数据增强
    full_dataset = LungNodule3DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=Transform3D())
    
    # 划分训练集与验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # 初始化 Lightning 模型
    lit_model = LitLungNoduleModel(learning_rate=1e-3)
    
    # 🛑 早停机制 (Early Stopping) —— 连续 10 个 epoch 验证集损失不降则停止
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    
    # 💾 自动保存最优权重 (Model Checkpoint)
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints/',
        filename='3d_resnet18-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        monitor='val_acc',
        mode='max'
    )
    
    # 📊 TensorBoard 实验日志记录器
    logger = TensorBoardLogger("models/logs/", name="baseline_3d_resnet18")
    
    # ⚙️ 实例化工业级工程训练器 Trainer
    trainer = pl.Trainer(
        max_epochs=30,                  # 最大迭代 30 轮
        accelerator='auto',             # 自动检测并调用可用显卡 (CUDA)
        devices=1,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )
    
    print("🚀 正在启动 PyTorch Lightning 3D-ResNet18 训练基线流水线...")
    trainer.fit(lit_model, train_loader, val_loader)