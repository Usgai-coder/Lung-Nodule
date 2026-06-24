import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# 导入 3D 数据集和 ResNet 模型
from dataset_3D import LungNodule3DDataset
from train_3D import LitLungNoduleModel

if __name__ == "__main__":
    MANIFEST_PATH = "data/processed/final_dataset_manifest.csv"
    CUBE_DIR = "data/processed/all_roi_cubes/"
    
    # ==========================================
    # ✂️ 【核心消融点】：关闭数据增强
    # ==========================================
    # 这里我们去掉了之前传入的 transform=Transform3D()
    # 让模型直接吃没有任何翻转、完全原始的数据，测试它会不会严重过拟合
    full_dataset = LungNodule3DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=None)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 强制固定随机种子，保证这里的划分和之前带有增强的实验一模一样，控制变量唯一
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # 初始化模型
    lit_model = LitLungNoduleModel(learning_rate=1e-3)
    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    
    # 将模型保存在专属的消融实验文件夹中
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints_ablation/',
        filename='3d_resnet18_no_aug-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1, monitor='val_acc', mode='max'
    )
    
    # 使用独立的日志名称，方便后续在 TensorBoard 里和之前的带增强版本直接对比
    logger = TensorBoardLogger("models/logs/", name="ablation_3d_resnet18_no_aug")
    
    trainer = pl.Trainer(
        max_epochs=30, accelerator='auto', devices=1, logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )
    
    print("🚀 启动消融实验：正在训练【无数据增强】版本的 3D-ResNet18 模型...")
    trainer.fit(lit_model, train_loader, val_loader)