import os
import glob
import torch
import numpy as np

# 导入核心网络模型
from train_3D_attention import LitAttentionModel

class LungNodulePredictor:
    """
    肺结节 3D 深度学习模型推理 API 接口
    支持单例预测 (predict_single) 和批量处理 (predict_batch)
    """
    def __init__(self, checkpoint_dir="models/checkpoints_attention/", device=None):
        """
        初始化推理引擎，自动挂载最优权重并分配硬件资源
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # 自动加载最新权重
        ckpts = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"未在 {checkpoint_dir} 找到模型权重。")
        ckpts.sort(key=os.path.getmtime)
        best_ckpt = ckpts[-1]
        
        print(f"⚙️ [API Init] 加载推理引擎模型: {os.path.basename(best_ckpt)}")
        print(f"⚙️ [API Init] 硬件加速器: {self.device}")
        
        self.model = LitAttentionModel.load_from_checkpoint(best_ckpt)
        self.model.to(self.device)
        self.model.eval() # 切换至推理模式
        
        self.classes = {0: "良性 (Benign)", 1: "恶性 (Malignant)"}

    def preprocess(self, npy_path):
        """
        前处理：加载 numpy 数组并转换为 (1, 1, 32, 32, 32) 的 Tensor 格式
        """
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"文件不存在: {npy_path}")
            
        cube = np.load(npy_path)
        if cube.shape != (32, 32, 32):
            raise ValueError(f"输入矩阵尺寸错误，期望 (32, 32, 32)，实际 {cube.shape}")
            
        # (D, H, W) -> (B, C, D, H, W) = (1, 1, 32, 32, 32)
        tensor = torch.tensor(cube, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_single(self, npy_path):
        """
        执行单例推理 (Single Inference)
        :return: dict 包含预测类别和恶意概率
        """
        input_tensor = self.preprocess(npy_path)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        pred_class_idx = np.argmax(probs)
        malignancy_prob = float(probs[1])
        
        return {
            "file": os.path.basename(npy_path),
            "prediction_label": self.classes[pred_class_idx],
            "malignancy_probability": round(malignancy_prob, 4),
            "is_high_risk": malignancy_prob > 0.5
        }

    @torch.no_grad()
    def predict_batch(self, npy_path_list):
        """
        执行批量推理 (Batch Inference)
        :return: list 包含多个推理结果的列表
        """
        results = []
        for path in npy_path_list:
            res = self.predict_single(path)
            results.append(res)
        return results

# ==========================================
# 🚀 API 使用示例 (Usage Example)
# ==========================================
if __name__ == "__main__":
    print("=== 🚀 [任务 8] 模型推理 API 接口测试 ===")
    try:
        # 初始化推理引擎
        predictor = LungNodulePredictor()
        
        # 寻找几个测试数据
        test_dir = "data/processed/all_roi_cubes/"
        sample_files = glob.glob(os.path.join(test_dir, "*.npy"))[:3]
        
        if len(sample_files) > 0:
            print("\n🔹 测试 1：单例推理 (predict_single)")
            single_res = predictor.predict_single(sample_files[0])
            print(single_res)
            
            print("\n🔹 测试 2：批量推理 (predict_batch)")
            batch_res = predictor.predict_batch(sample_files)
            for r in batch_res:
                print(f"  -> 文件: {r['file']}, 预测: {r['prediction_label']}, 恶性概率: {r['malignancy_probability']*100:.2f}%")
        else:
            print("未找到用于测试的 .npy 数据集，请检查路径。")
            
    except Exception as e:
        print(f"API 测试遇到错误: {e}")