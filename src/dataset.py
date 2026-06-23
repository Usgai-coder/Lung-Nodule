import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class LungNodule3DDataset(Dataset):
    """
    3D Lung Nodule Dataset for PyTorch (肺结节 3D 影像数据集)
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
        # 只保留文件名真正以 '.npy' 结尾的健康数据，自动踢掉所有写着"未提取或提取失败"的脏数据！
        self.manifest = raw_manifest[raw_manifest['cube_file_path'].str.endswith('.npy', na=False)].reset_index(drop=True)
        
    def __len__(self):
        """
        Returns the total number of samples (返回过滤后真正可用的健康样本总数).
        """
        return len(self.manifest)

    def __getitem__(self, idx):
        """
        Generates one sample of data (提取单个样本及其对应的标签).
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
        
        # 3. Add Channel dimension: (D, H, W) -> (C, D, H, W) (医学单通道灰度图 C=1)
        cube_array = np.expand_dims(cube_array, axis=0)
        
        # 4. Convert to PyTorch Tensor
        image_tensor = torch.tensor(cube_array, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 5. Apply transforms if any
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, label_tensor

# ==========================================
# 本地模块测试入口 (Local Module Test)
# ==========================================
if __name__ == "__main__":
    # 这里的路径请根据你的实际情况微调
    TEST_MANIFEST = "F:/Lung-Nodule/data/processed/final_dataset_manifest.csv"
    TEST_CUBE_DIR = "F:/Lung-Nodule/data/processed/all_roi_cubes/"
    
    try:
        # 初始化 Dataset
        test_dataset = LungNodule3DDataset(manifest_path=TEST_MANIFEST, cube_dir=TEST_CUBE_DIR)
        print(f"✅ Successfully loaded dataset with {len(test_dataset)} samples.")
        
        # 提取第一个样本测试形状
        img, lbl = test_dataset[0]
        print(f"📦 Image Tensor Shape: {img.shape}") # 期望输出: torch.Size([1, 32, 32, 32])
        print(f"🏷️ Label Tensor: {lbl}")
        
    except Exception as e:
        print(f"❌ Error during local testing: {e}")