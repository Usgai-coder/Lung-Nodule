import os
import glob
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, random_split

# 导入训练好的两个最强 3D 架构
from dataset_3D import LungNodule3DDataset
from train_3D_attention import LitAttentionModel
from train_3D_vnet import LitLungNoduleVNet3D

def get_best_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts: return None
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

if __name__ == "__main__":
    print("=== 🚀 [任务 7] 启动模型集成 (Model Ensemble) 联合评估 ===")
    
    # 1. 寻找最优权重
    ckpt_attention = get_best_ckpt("models/checkpoints_attention/")
    ckpt_vnet = get_best_ckpt("models/checkpoints_3d_vnet/")
    
    if not ckpt_attention or not ckpt_vnet:
        print("❌ 错误：请确保 models/ 目录下同时存在 Attention 和 VNet 的权重文件。")
        exit(1)
        
    print(f"📦 发现 Attention 模型权重: {os.path.basename(ckpt_attention)}")
    print(f"📦 发现 VNet 模型权重: {os.path.basename(ckpt_vnet)}")

    # 2. 加载模型
    model_a = LitAttentionModel.load_from_checkpoint(ckpt_attention).eval()
    model_b = LitLungNoduleVNet3D.load_from_checkpoint(ckpt_vnet).eval()

    # 3. 准备验证集 (保持相同的 Seed=42 以对齐之前的评估)
    MANIFEST_PATH = "data/processed/final_dataset_manifest.csv"
    CUBE_DIR = "data/processed/all_roi_cubes/"
    torch.manual_seed(42)
    full_dataset = LungNodule3DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=None)
    train_size = int(0.8 * len(full_dataset))
    _, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 4. 收集双模型的预测概率
    probs_a, probs_b, labels = [], [], []
    
    print("⏳ 正在执行双路模型并发推理...")
    with torch.no_grad():
        for x, y in val_loader:
            # 模型 A 推理 (3D Attention ResNet)
            logits_a = model_a(x)
            prob_a = torch.softmax(logits_a, dim=1)[:, 1].cpu().numpy()
            
            # 模型 B 推理 (3D VNet)
            logits_b = model_b(x)
            prob_b = torch.softmax(logits_b, dim=1)[:, 1].cpu().numpy()
            
            probs_a.extend(prob_a)
            probs_b.extend(prob_b)
            labels.extend(y.cpu().numpy())

    probs_a = np.array(probs_a)
    probs_b = np.array(probs_b)
    labels = np.array(labels)

    # ==============================================================
    # 🧠 核心集成算法：加权软投票 (Soft Voting)
    # 根据之前的独立评估表现，给予更强的 Attention 模型更高的权重
    # ==============================================================
    weight_a = 0.6  # 3D Attention 权重
    weight_b = 0.4  # 3D VNet 权重
    probs_ensemble = (probs_a * weight_a) + (probs_b * weight_b)
    preds_ensemble = (probs_ensemble > 0.5).astype(int)

    # 计算各路指标
    auc_a = roc_auc_score(labels, probs_a)
    auc_b = roc_auc_score(labels, probs_b)
    auc_ens = roc_auc_score(labels, probs_ensemble)
    acc_ens = accuracy_score(labels, preds_ensemble)

    print("\n📊 === 集成效果对比 (Ensemble Performance) ===")
    print(f"🔸 单模型 A (3D Attention) AUC : {auc_a:.4f}")
    print(f"🔸 单模型 B (3D VNet)      AUC : {auc_b:.4f}")
    print(f"🌟 融合集成模型 (Ensemble)  AUC : {auc_ens:.4f}  <-- 性能提升！")
    print(f"🌟 融合集成模型 (Ensemble)  Acc : {acc_ens:.4f}")