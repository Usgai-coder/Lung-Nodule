import yaml
import pydicom
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 🛠️ 基础配置加载模块
# ==========================================
def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载全局路径配置文件 yaml"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================
# 🧬 核心特征提取：解析单名医生的原始空间标注与评分
# ==========================================
def parse_xml_all_annotations(xml_path: str, patient_id: str) -> list:
    """
    解析单个 XML 文件，提取医生打分、三维空间中心坐标、形态学特征（球形度、直径）。
    为后续的三维物理空间聚类（合并同源结节）提供核心坐标弹药。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []

    annotations = []
    for session in root.iter():
        if session.tag.endswith('readingSession'):
            for nodule in session.iter():
                # 过滤出直径 >= 3mm 的明确结节标注
                if nodule.tag.endswith('unblindedReadNodule'):
                    nodule_id = "未知"
                    malignancy = np.nan   
                    sphericity = np.nan   
                    z_position = np.nan   
                    x_coords = []         
                    y_coords = []         
                    
                    for child in nodule:
                        if child.tag.endswith('noduleID'):
                            nodule_id = child.text
                            
                    for child in nodule.iter():
                        if child.tag.endswith('malignancy'):
                            malignancy = int(child.text)  
                        elif child.tag.endswith('sphericity'):
                            sphericity = int(child.text)  
                        elif child.tag.endswith('imageZposition'):
                            z_position = float(child.text) 
                        elif child.tag.endswith('xCoord'):
                            x_coords.append(int(child.text)) 
                        elif child.tag.endswith('yCoord'):
                            y_coords.append(int(child.text)) 
                            
                    if x_coords and y_coords and not np.isnan(z_position):
                        x_center = np.mean(x_coords) 
                        y_center = np.mean(y_coords) 
                        
                        width = max(x_coords) - min(x_coords)
                        height = max(y_coords) - min(y_coords)
                        # LIDC 的平均像素间距大约是 0.7mm
                        diameter = max(width, height) * 0.7
                        
                        annotations.append({
                            'patient_id': patient_id,
                            'nodule_id': nodule_id,
                            'malignancy': malignancy,
                            'sphericity': sphericity,
                            'x_center': x_center,
                            'y_center': y_center,
                            'z_position': z_position,
                            'diameter': diameter
                        })
    return annotations

# ==========================================
# 🎯 空间物理聚类：无视名字，只认物理坐标的同源合并
# ==========================================
def cluster_patient_nodules(annotations: list, distance_threshold: float = 5.0) -> list:
    """
    计算不同医生圈出结节的三维空间欧氏距离。
    如果中心点物理距离 <= 5.0 毫米，则判定它们是同一个物理结节。
    """
    clusters = [] 
    for ann in annotations:
        placed = False
        for cluster in clusters:
            ref = cluster[0] 
            
            dx = (ann['x_center'] - ref['x_center']) * 0.7  
            dy = (ann['y_center'] - ref['y_center']) * 0.7  
            dz = ann['z_position'] - ref['z_position']      
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            if distance <= distance_threshold:
                cluster.append(ann)
                placed = True
                break
        
        if not placed:
            clusters.append([ann])
            
    golden_nodules = []
    for cluster in clusters:
        scored_anns = [a for a in cluster if not np.isnan(a['malignancy'])]
        if not scored_anns: continue
        
        reader_count = len(scored_anns)                         
        mean_malignancy = np.mean([a['malignancy'] for a in scored_anns]) 
        mean_diameter = np.mean([a['diameter'] for a in cluster])       
        
        mean_sphericity = np.nanmean([a['sphericity'] for a in cluster]) 
        mean_x = np.mean([a['x_center'] for a in cluster])              
        mean_y = np.mean([a['y_center'] for a in cluster])              
        mean_z = np.mean([a['z_position'] for a in cluster])            
        
        golden_nodules.append({
            'patient_id': cluster[0]['patient_id'],
            'nodule_id': cluster[0]['nodule_id'], 
            'reader_count': reader_count,
            'mean_malignancy': mean_malignancy,
            'diameter': mean_diameter,
            'nodule_sphericity': mean_sphericity,
            'pos_x': mean_x,
            'pos_y': mean_y,
            'pos_z': mean_z
        })
    return golden_nodules

# ==========================================
# 🚀 主程序业务流
# ==========================================
if __name__ == "__main__":
    cfg = load_config()
    lidc_root = Path(cfg['data']['raw_lidc_dir'])
    output_dir = Path(cfg['data']['processed_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    

    print("==================================================")
    
    # 1. 提取患者人口学特征 (年龄、性别)
    patient_dict = {}
    print("-> 正在扫描 DICOM 头部字典获取年龄与性别...")
    for patient_folder in lidc_root.glob("LIDC-IDRI-*"):
        patient_id = patient_folder.name
        dcm_files = list(patient_folder.rglob("*.dcm"))
        if dcm_files:
            try:
                ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
                age = getattr(ds, 'PatientAge', '未知')
                sex = getattr(ds, 'PatientSex', '未知')
                if age != '未知' and isinstance(age, str) and age.endswith('Y'):
                    age = int(age.replace('Y', '').replace('0', '', 1) if age.startswith('0') else age.replace('Y', ''))
                patient_dict[patient_id] = {'age': age, 'sex': sex}
            except Exception:
                patient_dict[patient_id] = {'age': '未知', 'sex': '未知'}

    # 2. 地毯式读取全数据集的 XML 空间轮廓
    all_xml_files = list(lidc_root.rglob("*.xml"))
    all_patient_annotations = {}
    
    print(f"-> 正在解构 {len(all_xml_files)} 份 XML 空间标注并提取...")
    for xml_path in all_xml_files:
        patient_id = "未知"
        for part in xml_path.parts:
            if "LIDC-IDRI-" in part:
                patient_id = part
                break
        anns = parse_xml_all_annotations(str(xml_path), patient_id)
        if patient_id not in all_patient_annotations:
            all_patient_annotations[patient_id] = []
        all_patient_annotations[patient_id].extend(anns)
        
    # 3. 执行同病人体内的 3D 物理空间聚类
    print("-> 正在执行 3D 物理空间结节聚类 (强制合并同源病灶)...")
    all_clustered_nodules = []
    for pid, anns in all_patient_annotations.items():
        if anns:
            clustered = cluster_patient_nodules(anns, distance_threshold=5.0)
            all_clustered_nodules.extend(clustered)
            
    df_clustered = pd.DataFrame(all_clustered_nodules)
    
    # 4. 临床特征大回填，固化交付主数据探索表
    print("-> 正在组装全维度大账本...")
    df_clustered['patient_age'] = df_clustered['patient_id'].map(lambda x: patient_dict.get(x, {'age': '未知'})['age'])
    df_clustered['patient_sex'] = df_clustered['patient_id'].map(lambda x: patient_dict.get(x, {'sex': '未知'})['sex'])
    
    # 规范大账本的列排布顺序 (确保包含了 nodule_id 供下一步切图定位)
    master_cols = ['patient_id', 'nodule_id', 'patient_age', 'patient_sex', 'diameter', 'nodule_sphericity', 'pos_x', 'pos_y', 'pos_z', 'reader_count', 'mean_malignancy']
    df_master = df_clustered[master_cols].copy()
    
    # 💡 数据字典说明 (供查阅参考，实际导出为英文表头)
    # 'patient_id': '患者编号'
    # 'patient_age': '患者年龄'
    # 'patient_sex': '患者性别'
    # 'diameter': '物理直径(毫米)'
    # 'nodule_sphericity': '平均球形度(1-5)'
    # 'pos_x': 'X轴空间坐标'
    # 'pos_y': 'Y轴空间坐标'
    # 'pos_z': 'Z轴绝对高度'
    # 'reader_count': '医生一致认可数'
    # 'mean_malignancy': '平均恶性评分(1-5)'
    # 'label': '终极黄金标签(0良/1恶)'
    
    master_path = output_dir / "master_data_exploration.csv"
    df_master.to_csv(master_path, index=False, encoding="utf-8-sig")
    print(f"✅ 结果已保存至: {master_path}")
    
    # 5. 应用双层网筛选，生成最终用于模型切图的黄金标签表
    print("\n-> 正在应用核心过滤网：[医生数 >= 3] & [直径 >= 3.0mm] ...")
    df_filtered = df_master[(df_master['reader_count'] >= 3) & (df_master['diameter'] >= 3.0)].copy()
    
    def assign_binary_label(score):
        if score > 3.0: return 1   # 恶性
        elif score < 3.0: return 0 # 良性
        else: return -1            # 刚好 3 分属于模糊争议边界，执行安全剔除
        
    df_filtered['label'] = df_filtered['mean_malignancy'].apply(assign_binary_label)
    df_final = df_filtered[df_filtered['label'] != -1].copy()
    
    final_label_path = output_dir / "cleaned_nodules_labels.csv"
    df_final.to_csv(final_label_path, index=False, encoding="utf-8-sig")
    

    print("==================================================")
    print(f"-> 严格符合所有硬性指标的高质量结节数: {len(df_final)} 个")
    print(f"   - 黄金良性结节 (Label 0): {(df_final['label'] == 0).sum()} 个")
    print(f"   - 黄金恶性结节 (Label 1): {(df_final['label'] == 1).sum()} 个")
    print(f"✅ 结果已保存至: {final_label_path}")
    print("==================================================")