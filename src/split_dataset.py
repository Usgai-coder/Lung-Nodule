import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 🛠️ 基础配置加载
# ==========================================
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================
# 🚀 第八步：终极硬核 - 手动强制保底分层算法
# ==========================================
if __name__ == "__main__":
    cfg = load_config()
    processed_dir = Path(cfg['data']['processed_dir'])
    
    manifest_path = processed_dir / "final_dataset_manifest.csv"
    if not manifest_path.exists():
        print(f"【严重错误】找不到全数据标签账本：{manifest_path}")
        exit()
        
    df = pd.read_csv(manifest_path)
    
    print("==================================================")
    print("⚖️ 启动极端不平衡数据的手动强制保底发牌...")
    
    # 1. 提取患者级最高风险标签
    patient_df = df.groupby('patient_id')['multi_label'].max().reset_index()
    patient_df.rename(columns={'multi_label': 'patient_max_risk'}, inplace=True)
    print(f"-> 总独立患者数: {len(patient_df)} 位")
    
    # 2. 手动医疗级分层发牌算法
    patient_to_split = {}
    np.random.seed(42) # 锁定随机种子，保证每次运行切分结果绝对一致
    
    for risk_label, group in patient_df.groupby('patient_max_risk'):
        patients = group['patient_id'].values.copy()
        np.random.shuffle(patients) # 打乱该类别的患者
        
        n = len(patients)
        
        # 💡 核心魔法：如果是有病灶的患者(risk > 0) 且总人数>=3，强制保底分发！
        if risk_label > 0 and n >= 3:
            test_count = max(1, int(np.round(n * 0.2)))  # 至少保底 1 人进测试集
            val_count = max(1, int(np.round(n * 0.1)))   # 至少保底 1 人进验证集
        else:
            # 健康对照组（几百上千人），按正常数学比例随意切
            test_count = int(np.round(n * 0.2))
            val_count = int(np.round(n * 0.1))
            
        test_pts = patients[:test_count]
        val_pts = patients[test_count : test_count + val_count]
        train_pts = patients[test_count + val_count :]
        
        for p in test_pts: patient_to_split[p] = 'test'
        for p in val_pts:  patient_to_split[p] = 'val'
        for p in train_pts: patient_to_split[p] = 'train'
    
    # 3. 映射回原表并保存
    df['split'] = df['patient_id'].map(patient_to_split)
    final_split_path = processed_dir / "final_split_train_manifest.csv"
    df.to_csv(final_split_path, index=False, encoding="utf-8-sig")
    
    print("\n==================================================")
    print("📊 强制保底数据集划分（7:1:2）大盘统计报告")
    print("==================================================")
    print("📌 [1] 原始样本阵营分布:")
    print(df['split'].value_counts().to_string())
    
    print("\n☯️ [2] 二分类(良/恶)在各阵营的交叉分布:")
    print(pd.crosstab(df['split'], df['binary_desc']))
    
    print("\n🔮 [3] 多分类(低/中/高风险)在各阵营的交叉分布:")
    print(pd.crosstab(df['split'], df['multi_desc']))
    print("==================================================")