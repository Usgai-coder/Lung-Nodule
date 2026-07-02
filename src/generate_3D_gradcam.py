import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset_3D import LungNodule3DDataset
from train_3D_attention import LitAttentionModel  

# ==========================================
# 🧠 1. 核心算法：自定义 3D Grad-CAM
# ==========================================
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子截获特征图和梯度
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, target_class]
        
        # 修复：去掉 retain_graph=True，防止循环处理多个图像时显存泄漏或计算图崩溃
        score.backward() 
        
        weights = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
        
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

# ==========================================
# 🎨 2. 可视化模块：中心切片叠加渲染
# ==========================================
def show_cam_on_image(img_2d, mask_2d, save_path, title_prefix="Malignant"):
    img_2d_uint8 = np.uint8(255 * img_2d)
    img_color = cv2.cvtColor(img_2d_uint8, cv2.COLOR_GRAY2RGB)
    
    heatmap = np.uint8(255 * mask_2d)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 🌟 修复核心：将 OpenCV 的 BGR 格式转换为 matplotlib 需要的 RGB 格式
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    cam_img = cv2.addWeighted(heatmap, 0.4, img_color, 0.6, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_2d_uint8, cmap='gray')
    axes[0].set_title('Original CT Slice', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    # 🌟 满足要求：明确标注热力图对准的目标类别
    axes[1].set_title(f'Grad-CAM\n(Target: {title_prefix})', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(cam_img)
    # 🌟 满足要求：增加红色代表高注意力的说明
    axes[2].set_title('Overlay\n(Red = High Attention)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# 🚀 3. 执行脚本
# ==========================================
if __name__ == "__main__":
    out_dir = "models/evaluation_results/grad_cam/"
    os.makedirs(out_dir, exist_ok=True)
    
    ckpt_dir = "models/checkpoints_attention/"
    ckpts = glob.glob(os.path.join(ckpt_dir, "*cbam*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("❌ 未找到模型权重，请检查路径！")
    
    ckpts.sort(key=os.path.getmtime)
    best_ckpt = ckpts[-1]
    print(f"📦 正在加载最优权重: {os.path.basename(best_ckpt)}")
    
    lit_model = LitAttentionModel.load_from_checkpoint(best_ckpt)
    lit_model.eval()
    
    target_layer = lit_model.model.layer2[-1]
    grad_cam = GradCAM3D(lit_model.model, target_layer)
    
    MANIFEST_PATH = "data/processed/final_dataset_manifest.csv"
    CUBE_DIR = "data/processed/all_roi_cubes/"
    dataset = LungNodule3DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=None)
    
    print("⏳ 正在搜寻典型样本生成热力图 (目标: 2个恶性, 2个良性)...")
    
    malignant_count = 0
    benign_count = 0
    
    for idx in range(len(dataset)):
        img_tensor, label = dataset[idx]
        is_malignant = (label.item() == 1)
        
        # 已经找够了就跳过
        if is_malignant and malignant_count >= 2:
            continue
        if not is_malignant and benign_count >= 2:
            continue
            
        print(f"🔍 扫描到第 {idx} 个样本，类别: {'恶性' if is_malignant else '良性'}")
        input_tensor = img_tensor.unsqueeze(0)
        
        # 生成对应的 3D 热力图
        cam_3d = grad_cam.generate(input_tensor, target_class=label.item())
        
        center_z = input_tensor.shape[2] // 2
        img_slice = input_tensor[0, 0, center_z, :, :].numpy()
        cam_slice = cam_3d[center_z, :, :]
        
        if is_malignant:
            malignant_count += 1
            save_name = os.path.join(out_dir, f"CAM_Malignant_{malignant_count}.png")
            show_cam_on_image(img_slice, cam_slice, save_name, title_prefix="Malignant")
            print(f"✅ 生成 [恶性] 对比图 -> {save_name}")
        else:
            benign_count += 1
            save_name = os.path.join(out_dir, f"CAM_Benign_{benign_count}.png")
            show_cam_on_image(img_slice, cam_slice, save_name, title_prefix="Benign")
            print(f"✅ 生成 [良性] 对比图 -> {save_name}")
            
        if malignant_count >= 2 and benign_count >= 2:
            break
                
    print(f"\n🎉 Grad-CAM 3D 解释性分析完成！共生成 4 张图，保存在 {out_dir} ")