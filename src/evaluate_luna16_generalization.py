import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from torch.utils.data import DataLoader

# ==========================================
# 📦 导入所有 3D 与 2D 数据集和模型封装
# ==========================================
from dataset_3D import LungNodule3DDataset
from train_3D import LitLungNoduleModel          # 3D ResNet
from train_3D_vnet import LitLungNoduleVNet3D    # 3D VNet

from dataset_2D import LungNodule2DDataset
from train_2D import LitLungNoduleModel2D        # 2D ResNet 
from train_2D_densenet import LitLungNoduleDenseNet2D # 2D DenseNet

def get_best_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

def evaluate_model(model, val_loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def print_metrics(name, y_true, y_pred, y_probs):
    print(f"\n--- {name} LUNA16 泛化评估指标 ---")
    print(f"💡 准确率 (Accuracy)  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"🎯 精确率 (Precision) : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"🔍 召回率 (Recall)    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"⚖️ F1 分数 (F1-Score) : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"📈 AUC-ROC 指标       : {roc_auc_score(y_true, y_probs):.4f}")

if __name__ == "__main__":
    print("=== 🚀 启动 LUNA16 跨数据集泛化能力验证 (包含2D与3D) ===")
    out_dir = "models/evaluation_results/luna16_generalization/"
    os.makedirs(out_dir, exist_ok=True)
    
    LUNA16_MANIFEST = "data/processed/luna16_dataset_manifest.csv"
    LUNA16_CUBE_DIR = "data/processed/luna16_roi_cubes/"
    
    if not os.path.exists(LUNA16_MANIFEST):
        print(f"\n❌ 找不到 LUNA16 账本文件: {LUNA16_MANIFEST}")
        exit(1)
        
    # ==========================================
    # 🗄️ 准备 LUNA16 的 3D 与 2D 测试集加载器
    # ==========================================
    torch.manual_seed(42)
    luna16_dataset_3d = LungNodule3DDataset(manifest_path=LUNA16_MANIFEST, cube_dir=LUNA16_CUBE_DIR, transform=None)
    luna16_loader_3d = DataLoader(luna16_dataset_3d, batch_size=8, shuffle=False, num_workers=0)
    
    torch.manual_seed(42)
    luna16_dataset_2d = LungNodule2DDataset(manifest_path=LUNA16_MANIFEST, cube_dir=LUNA16_CUBE_DIR, transform=None)
    luna16_loader_2d = DataLoader(luna16_dataset_2d, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"✅ 成功加载 LUNA16 独立测试集，共计 {len(luna16_dataset_3d)} 个样本")
    
    results_3d = {}
    results_2d = {}
    
    # ==========================================
    # 🌟 评估 3D 模型组
    # ==========================================
    ckpt_resnet = get_best_ckpt("models/checkpoints/")
    if ckpt_resnet:
        print(f"\n📦 加载 3D ResNet 最优权重进行泛化测试: {os.path.basename(ckpt_resnet)}")
        model_resnet = LitLungNoduleModel.load_from_checkpoint(ckpt_resnet)
        y_true, y_pred, y_probs = evaluate_model(model_resnet, luna16_loader_3d)
        results_3d['ResNet3D'] = (y_true, y_pred, y_probs)
        print_metrics('3D ResNet18 (LUNA16)', y_true, y_pred, y_probs)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Reds', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title('LUNA16 Generalization CM (3D ResNet18)')
        plt.savefig(os.path.join(out_dir, 'luna16_3d_resnet_cm.png'), dpi=300)
        plt.close()

    ckpt_vnet = get_best_ckpt("models/checkpoints_3d_vnet/")
    if ckpt_vnet:
        print(f"\n📦 加载 3D VNet 最优权重进行泛化测试: {os.path.basename(ckpt_vnet)}")
        model_vnet = LitLungNoduleVNet3D.load_from_checkpoint(ckpt_vnet)
        y_true, y_pred, y_probs = evaluate_model(model_vnet, luna16_loader_3d)
        results_3d['VNet3D'] = (y_true, y_pred, y_probs)
        print_metrics('3D VNet (LUNA16)', y_true, y_pred, y_probs)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Greens', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title('LUNA16 Generalization CM (3D VNet)')
        plt.savefig(os.path.join(out_dir, 'luna16_3d_vnet_cm.png'), dpi=300)
        plt.close()

    # ==========================================
    # 🌟 评估 2D 模型组
    # ==========================================
    ckpt_resnet_2d = get_best_ckpt("models/checkpoints_2d/")
    if ckpt_resnet_2d:
        print(f"\n📦 加载 2D ResNet 最优权重进行泛化测试: {os.path.basename(ckpt_resnet_2d)}")
        # 修正: 使用正确的类名 LitLungNoduleModel2D 实例化
        model_resnet_2d = LitLungNoduleModel2D.load_from_checkpoint(ckpt_resnet_2d)
        y_true, y_pred, y_probs = evaluate_model(model_resnet_2d, luna16_loader_2d)
        results_2d['ResNet2D'] = (y_true, y_pred, y_probs)
        print_metrics('2D ResNet18 (LUNA16)', y_true, y_pred, y_probs)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Oranges', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title('LUNA16 Generalization CM (2D ResNet18)')
        plt.savefig(os.path.join(out_dir, 'luna16_2d_resnet_cm.png'), dpi=300)
        plt.close()

    ckpt_densenet_2d = get_best_ckpt("models/checkpoints_2d_densenet/")
    if ckpt_densenet_2d:
        print(f"\n📦 加载 2D DenseNet 最优权重进行泛化测试: {os.path.basename(ckpt_densenet_2d)}")
        model_densenet_2d = LitLungNoduleDenseNet2D.load_from_checkpoint(ckpt_densenet_2d)
        y_true, y_pred, y_probs = evaluate_model(model_densenet_2d, luna16_loader_2d)
        results_2d['DenseNet2D'] = (y_true, y_pred, y_probs)
        print_metrics('2D DenseNet121 (LUNA16)', y_true, y_pred, y_probs)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Purples', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title('LUNA16 Generalization CM (2D DenseNet121)')
        plt.savefig(os.path.join(out_dir, 'luna16_2d_densenet_cm.png'), dpi=300)
        plt.close()

    # ==========================================
    # 📈 绘制 LUNA16 泛化能力 ROC 对比曲线
    # ==========================================
    # 绘制 3D 组
    if len(results_3d) > 0:
        plt.figure(figsize=(7, 6))
        colors_3d = {'ResNet3D': 'deeppink', 'VNet3D': 'forestgreen'}
        for name, (y_t, y_p, y_prob) in results_3d.items():
            fpr, tpr, _ = roc_curve(y_t, y_prob)
            auc = roc_auc_score(y_t, y_prob)
            plt.plot(fpr, tpr, color=colors_3d.get(name, 'black'), lw=2, label=f'{name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)'); plt.ylabel('True Positive Rate (TPR)')
        plt.title('LUNA16 Generalization Test (3D Models)')
        plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'luna16_3d_generalization_roc.png'), dpi=300)
        plt.close()
        print(f"\n✅ 3D 模型 LUNA16 泛化 ROC 曲线已生成！")

    # 绘制 2D 组
    if len(results_2d) > 0:
        plt.figure(figsize=(7, 6))
        colors_2d = {'ResNet2D': 'darkorange', 'DenseNet2D': 'purple'}
        for name, (y_t, y_p, y_prob) in results_2d.items():
            fpr, tpr, _ = roc_curve(y_t, y_prob)
            auc = roc_auc_score(y_t, y_prob)
            plt.plot(fpr, tpr, color=colors_2d.get(name, 'black'), lw=2, label=f'{name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)'); plt.ylabel('True Positive Rate (TPR)')
        plt.title('LUNA16 Generalization Test (2D Models)')
        plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'luna16_2d_generalization_roc.png'), dpi=300)
        plt.close()
        print(f"✅ 2D 模型 LUNA16 泛化 ROC 曲线已生成！")
        
    print(f"\n🎉 恭喜！所有测试图表已完整保存至: {out_dir}")