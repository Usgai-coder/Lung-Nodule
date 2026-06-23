import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class LungNodule2DDataset(Dataset):
    """
    2D Lung Nodule Dataset for PyTorch (肺结节 2D 横截面影像数据集)
    """
    def __init__(self, manifest_path, cube_dir, transform=None):
        """
        Args:
            manifest_path (str): 索引表路径 (Path to the manifest CSV file).
            cube_dir (str): 3D ROI 矩阵所在的物理文件夹 (Directory containing .npy cubes).
            transform (callable, optional): 数据增强模块 (Optional transforms to be applied).
        """
        self.cube_dir = cube_dir
        self.transform = transform
        
        # 1. 读入原始的 CSV 资产总账
        raw_manifest = pd.read_csv(manifest_path)
        
        # 2. 【核心修改：强力过滤网】
        # 只保留文件名真正以 '.npy' 结尾的健康数据，自动踢掉所有写着"未提取或提取失败"的数据
        self.manifest = raw_manifest[raw_manifest['cube_file_path'].str.endswith('.npy', na=False)].reset_index(drop=True)
        
    def __len__(self):
        """
        Returns the total number of samples (返回过滤后真正可用的健康样本总数).
        """
        return len(self.manifest)

    def _get_max_cross_section(self, cube_array):
        """
        核心算法：寻找结节的最大横截面 (Max Cross-Section)
        遍历 Z 轴的每一层切片，计算每一层非背景像素的数量，找出面积最大的那一层。
        """
        max_area = -1
        best_slice_idx = cube_array.shape[0] // 2  # 设定默认值为中心层
        
        for z in range(cube_array.shape[0]):
            slice_2d = cube_array[z, :, :]
            area = np.sum(slice_2d > 0.01) 
            if area > max_area:
                max_area = area
                best_slice_idx = z
                
        return cube_array[best_slice_idx, :, :]

    def __getitem__(self, idx):
        """
        Generates one sample of data (提取单个样本的 2D 最大横截面及其对应的标签).
        """
        # 1. Get file name and label from the manifest
        row = self.manifest.iloc[idx]
        file_name = row['cube_file_path'] 
        
        # 将中文标签转换为 PyTorch 认识的数字 (0: 良性, 1: 恶性)
        label_str = str(row['binary_desc'])
        if '良性' in label_str:
            label = 0
        else:
            label = 1
        
        # 2. Load the 3D numpy array (读取 32x32x32 的局部 3D 矩阵)
        pure_file_name = os.path.basename(file_name)
        cube_path = os.path.join(self.cube_dir, pure_file_name)
        cube_array = np.load(cube_path)
        
        # 3. 🌟 压缩提取最大横截面 (D, H, W) -> (H, W)
        slice_2d = self._get_max_cross_section(cube_array)
        
        # 4. Add Channel dimension: (H, W) -> (C, H, W) (医学单通道灰度图 C=1)
        slice_2d = np.expand_dims(slice_2d, axis=0)
        
        # 5. Convert to PyTorch Tensor
        image_tensor = torch.tensor(slice_2d, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 6. Apply transforms if any
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, label_tensor

# ==========================================
# 本地模块测试入口 (Local Module Test)
# ==========================================
if __name__ == "__main__":
    # 这里的路径请根据实际情况微调
    TEST_MANIFEST = "F:/Lung-Nodule/data/processed/final_dataset_manifest.csv"
    TEST_CUBE_DIR = "F:/Lung-Nodule/data/processed/all_roi_cubes/"
    
    try:
        # 初始化 Dataset
        test_dataset = LungNodule2DDataset(manifest_path=TEST_MANIFEST, cube_dir=TEST_CUBE_DIR)
        print(f"✅ Successfully loaded dataset with {len(test_dataset)} samples.")
        
        # 提取第一个样本测试形状
        img, lbl = test_dataset[0]
        print(f"📦 Image Tensor Shape: {img.shape}") # 期望输出: torch.Size([1, 32, 32])
        print(f"🏷️ Label Tensor: {lbl}")
        
    except Exception as e:
        print(f"❌ Error during local testing: {e}")