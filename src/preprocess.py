import yaml
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import os

# 屏蔽 ITK 底层警告，保持输出整洁
sitk.ProcessObject_SetGlobalWarningDisplay(False)

# ==========================================
# 🛠️ 功能 0：基础配置加载
# ==========================================
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================
# 🛠️ 功能 1：去标识化 (De-identification)
# ==========================================
def read_3d_image(file_path: str):
    """
    SimpleITK 在读取 LUNA16 .mhd 图像时，会自动剥离元数据字典中包含的隐私信息，
    向外只输出纯粹的空间几何矩阵，天然打通了临床数据的【去标识化】。
    """
    return sitk.ReadImage(file_path)

# ==========================================
# 🛠️ 功能 2：3D 空间重采样 (Resampling)
# ==========================================
def resample_image(itk_image, new_spacing=[1.0, 1.0, 1.0]):
    """
    利用三维线性插值，将原始厚薄不一的 CT 层厚（如 Z 轴 2.5mm），
    各向同性化压缩统一为 1mm x 1mm x 1mm 的完美正方体，消除空间形变。
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    new_size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(-1000) # 用空气的绝对低值填充外部背景
    resample.SetInterpolator(sitk.sitkLinear)
    
    return resample.Execute(itk_image)

# ==========================================
# 🛠️ 功能 3：硬核 3D 肺实质分割 (Lung Segmentation)
# ==========================================
def segment_lung_mask(itk_image):
    """
    利用数学形态学算法，精准剔除胸壁和扫描床，提取双肺闭合区域。
    1. 自动二值化：肺部区域 HU 值通常在 -400 以下。
    2. 3D 形态学闭运算：使用球形结构元，自动填补肺内大血管引起的空洞，
       并防止紧贴在肺壁边缘长出来的结节被错误切掉。
    """
    # 阈值二值化（得到粗糙的肺部气腔掩膜）
    binary_mask = sitk.BinaryThreshold(itk_image, lowerThreshold=-1000, upperThreshold=-400, insideValue=1, outsideValue=0)
    
    # 配置三维球形膨胀腐蚀结构元 (3D Ball Kernel) 执行闭运算
    closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
    closing_filter.SetKernelRadius([3, 3, 3]) # 3像素半径的三维球体
    closing_filter.SetKernelType(sitk.sitkBall)
    
    cleaned_mask = closing_filter.Execute(binary_mask)
    return cleaned_mask

# ==========================================
# 🛠️ 功能 4：HU 值归一化 (HU Normalization)
# ==========================================
def normalize_hu(image_array: np.ndarray, min_bound: float = -1000.0, max_bound: float = 400.0) -> np.ndarray:
    """
    将亨氏单位（HU）安全截断在有效窗宽 [-1000, 400] 内，
    并线性缩放到深度学习框架最喜爱的 [0.0, 1.0] 的 float32 空间。
    """
    image_array = np.clip(image_array, min_bound, max_bound)
    image_array = (image_array - min_bound) / (max_bound - min_bound)
    return image_array.astype(np.float32)

# ==========================================
# 🚀 全自动化预处理
# ==========================================
if __name__ == "__main__":
    cfg = load_config()
    luna_root = Path(cfg['data']['raw_luna_dir'])
    
    # 定义预处理后矩阵的保存目录，不存在则自动创建
    output_dir = Path(cfg['data']['processed_dir']) / "preprocessed_cubes"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("正在扫描并加载 LUNA16 数据集...")
    mhd_files = list(luna_root.rglob("*.mhd"))
    
    if not mhd_files:
        print(f"【错误】未在 {luna_root} 找到 .mhd 文件。")
    else:
        print("==================================================")
        print(f"🎬 共检索到 {len(mhd_files)} 例原始影像")
        print(f"💾 输出目标文件夹: {output_dir}")
        print("==================================================")
        
        success_count = 0
        
        for idx, file_path in enumerate(mhd_files):
            # 提取病例的唯一 UID 作为保存的文件名
            case_uid = file_path.stem
            save_path = output_dir / f"{case_uid}_preprocessed.npy"
            
            # 如果之前已经处理过，直接跳过，支持断点续传
            if save_path.exists():
                print(f"⏩ 进度 [{idx+1}/{len(mhd_files)}] 发现已存在的缓存，直接跳过: {case_uid[:20]}...")
                success_count += 1
                continue
                
            print(f"⚙️ 进度 [{idx+1}/{len(mhd_files)}] 正在全力加工: {case_uid[:20]}...")
            
            try:
                # 1. 读取原图
                original_image = read_3d_image(str(file_path))
                
                # 2. 空间重采样
                resampled_image = resample_image(original_image, new_spacing=[1.0, 1.0, 1.0])
                
                # 3. 肺实质分割
                lung_mask_image = segment_lung_mask(resampled_image)
                
                # 4. 转为矩阵并利用掩膜去噪 
                img_array = sitk.GetArrayFromImage(resampled_image)
                mask_array = sitk.GetArrayFromImage(lung_mask_image)
                
                # 用空气(-1000)填充非肺部区域
                segmented_lung_array = np.where(mask_array == 1, img_array, -1000)
                
                # 5. HU 值归一化
                final_array = normalize_hu(segmented_lung_array)
                
                # 6. 保存为 Numpy 矩阵
                np.save(save_path, final_array)
                success_count += 1
                
            except Exception as e:
                print(f"  -> ⚠️ 病例 {case_uid[:15]} 加工失败，跳过。原因: {e}")
                
        print("\n==================================================")
        print(f"成功将 {success_count}/{len(mhd_files)} 例 3D 影像输出")
        print(f"结果已安全保存至: {output_dir}")
        print("==================================================")