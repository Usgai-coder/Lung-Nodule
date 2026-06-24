import os
import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk

def extract_luna16_rois():
    print("=== 🚀 启动 LUNA16 物理空间 ROI 裁切与账本生成 ===")
    
    # ==========================================
    # 📁 1. 路径配置 (请确保这些路径与你本地一致)
    # ==========================================
    # LUNA16 官方提供的标注文件 (包含结节坐标和直径)
    annotations_path = "data/LUNA16/annotations.csv" 
    # 原始数据文件夹 (用于读取 mhd 的物理原点 metadata)
    raw_luna_dir = "data/LUNA16/"
    # preprocess.py 跑出来的全肺归一化矩阵存放地
    preprocessed_dir = "data/processed/preprocessed_cubes/"
    
    # 输出路径
    out_cube_dir = "data/processed/luna16_roi_cubes/"
    out_manifest_path = "data/processed/luna16_dataset_manifest.csv"
    
    os.makedirs(out_cube_dir, exist_ok=True)
    
    if not os.path.exists(annotations_path):
        print(f"\n❌ 严重错误: 找不到 LUNA16 官方标注文件: {annotations_path}")
        print("请确保下载了 LUNA16 的 annotations.csv 并放在 data/LUNA16/ 目录下")
        return

    # ==========================================
    # 📖 2. 读取 LUNA16 标注并开始处理
    # ==========================================
    annotations = pd.read_csv(annotations_path)
    print(f"✅ 成功加载 annotations.csv，共发现 {len(annotations)} 个官方标注结节")
    
    manifest_records = []
    success_count = 0
    missing_files = set()

    for idx, row in annotations.iterrows():
        uid = str(row['seriesuid'])
        coord_x = float(row['coordX'])
        coord_y = float(row['coordY'])
        coord_z = float(row['coordZ'])
        diameter = float(row['diameter_mm'])
        
        # 寻找对应的全肺矩阵文件
        npy_path = os.path.join(preprocessed_dir, f"{uid}_preprocessed.npy")
        if not os.path.exists(npy_path):
            missing_files.add(uid)
            continue
            
        # 寻找对应的原始 mhd 文件 (为了获取物理空间的 Origin 和 Direction)
        # 因为我们预处理时把 spacing 变成了 1x1x1，但 Origin 没变
        mhd_files = glob.glob(os.path.join(raw_luna_dir, "**", f"{uid}.mhd"), recursive=True)
        if not mhd_files:
            missing_files.add(uid)
            continue
            
        mhd_path = mhd_files[0]
        
        # ==========================================
        # 📐 3. 物理坐标 -> 矩阵索引映射 (核心算法)
        # ==========================================
        try:
            # 高效读取 mhd 的元数据 (只读头文件，不读庞大的像素数据)
            reader = sitk.ImageFileReader()
            reader.SetFileName(mhd_path)
            reader.ReadImageInformation()
            origin = reader.GetOrigin()
            direction = reader.GetDirection()
            
            # 读取预处理好的全肺 numpy 矩阵
            lung_array = np.load(npy_path)
            D, H, W = lung_array.shape
            
            # 创建一个虚拟的 SimpleITK 图像，用于调用自带的物理映射算子
            # 注意：sitk 的 Size 是 [X, Y, Z]，对应 Numpy 的 [W, H, D]
            dummy_img = sitk.Image([W, H, D], sitk.sitkUInt8)
            dummy_img.SetOrigin(origin)
            dummy_img.SetDirection(direction)
            dummy_img.SetSpacing([1.0, 1.0, 1.0]) # 预处理时我们强行缩放到了 1x1x1
            
            # 将物理坐标映射为矩阵索引
            physical_point = (coord_x, coord_y, coord_z)
            idx_x, idx_y, idx_z = dummy_img.TransformPhysicalPointToIndex(physical_point)
            
            # 转为 Numpy 的 (Depth, Height, Width) 顺序
            center_d, center_h, center_w = idx_z, idx_y, idx_x
            
            # ==========================================
            # ✂️ 4. 裁切 32x32x32 局部 ROI (含防越界保护)
            # ==========================================
            half_size = 16
            d_start, d_end = center_d - half_size, center_d + half_size
            h_start, h_end = center_h - half_size, center_h + half_size
            w_start, w_end = center_w - half_size, center_w + half_size
            
            # 预处理时归一化 [-1000, 400] -> [0, 1]。空气 -1000 对应的值是 0.0
            # 因此使用 0.0 作为 padding 背景值
            roi_cube = np.zeros((32, 32, 32), dtype=np.float32)
            
            # 计算合法的截取边界
            d_start_v, d_end_v = max(0, d_start), min(D, d_end)
            h_start_v, h_end_v = max(0, h_start), min(H, h_end)
            w_start_v, w_end_v = max(0, w_start), min(W, w_end)
            
            # 如果中心点完全在体外，跳过
            if d_start_v >= d_end_v or h_start_v >= h_end_v or w_start_v >= w_end_v:
                continue
                
            # 计算在 roi_cube 中对应的粘贴位置
            roi_d_start = d_start_v - d_start
            roi_d_end = 32 - (d_end - d_end_v)
            roi_h_start = h_start_v - h_start
            roi_h_end = 32 - (h_end - h_end_v)
            roi_w_start = w_start_v - w_start
            roi_w_end = 32 - (w_end - w_end_v)
            
            # 复制切片数据
            roi_cube[roi_d_start:roi_d_end, roi_h_start:roi_h_end, roi_w_start:roi_w_end] = \
                lung_array[d_start_v:d_end_v, h_start_v:h_end_v, w_start_v:w_end_v]
            
            # ==========================================
            # 💾 5. 生成伪标签并保存
            # ==========================================
            # 临床规则：直径 >= 8mm 视为高风险/恶性 (1)，否则为良性 (0)
            binary_label = 1 if diameter >= 8.0 else 0
            binary_desc = "恶性结节" if binary_label == 1 else "良性结节"
            
            roi_filename = f"{uid}_nodule_{idx}.npy"
            roi_save_path = os.path.join(out_cube_dir, roi_filename)
            
            np.save(roi_save_path, roi_cube)
            
            # 记录到账本中 (字段名需与你 dataset.py 读取的列名对齐)
            manifest_records.append({
                "patient_id": uid,
                "cube_file_path": roi_filename,
                "diameter_mm": diameter,
                "binary_label": binary_label,
                "binary_desc": binary_desc
            })
            
            success_count += 1
            if success_count % 50 == 0:
                print(f"  -> 已成功处理 {success_count} 个结节...")
                
        except Exception as e:
            print(f"⚠️ 处理病灶 {uid} 时发生异常: {e}")
            continue

    # ==========================================
    # 📝 6. 保存最终大账本
    # ==========================================
    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(out_manifest_path, index=False, encoding='utf-8-sig')
    
    print("\n==================================================")
    print(f"🎉 LUNA16 ROI 裁切完毕")
    print(f"✅ 成功裁切并生成 {success_count} 个 3D ROI 矩阵")
    print(f"📄 泛化测试账本已生成: {out_manifest_path}")
    if missing_files:
        print(f"⚠️ 有 {len(missing_files)} 个 UID 缺失对应的 mhd 或 npy 文件，已跳过")
    print("==================================================")

if __name__ == "__main__":
    extract_luna16_rois()