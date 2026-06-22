# 🫁 肺结节 3D 影像标准化预处理与多维特征构建系统 (LIDC-IDRI / LUNA16)

---

## * 1. 项目背景与核心学术贡献 (Project Highlights)
本项目基于国际权威的 **LIDC-IDRI** 及其衍生基准 **LUNA16**，打造了一套从原始异构医疗数据（DICOM/MHD + XML）到深度学习标准张量的端到端全自动预处理流水线。

项目在工程与算法层面，核心突破了以下四大行业痛点：
* **各向同性重采样与肺实质精准分割**：
  针对 CT 扫描 Z 轴层厚不均（如 2.5mm）引发的 3D 卷积形变，强制执行 1 x 1 x 1 mm 的等方性三维线性插值。采用纯 `SimpleITK` 构建三维形态学闭运算管道，精准剥离胸壁与扫描床。摒弃常规掩膜的“乘零”错误，引入条件映射（`np.where`）将肺外背景严格还原为代表空气的 -1000 HU，确保模型物理语义的绝对正确。
* **阅片者间变异性消除**：
  引入 **3D 物理空间欧氏距离聚类算法**，强制合并多位医生在空间距离 <= 5.0 mm 内圈定的同源病灶，生成零争议的“黄金结节”大账本。
* **三维局部裁切与归一化**：
  在三维绝对物理坐标系下精准定位，裁切 32 x 32 x 32 体素的局部立方体，并完成肺窗（[-1000, 400]）的绝对线性归一化。
* **严格患者级防泄露切分**：
  采用 **Patient-Level 均衡发牌算法**，根据患者体内的最大风险特征进行分层摇号，以 **7:1:2** 的完美比例锁定划分，从根源上杜绝深度学习中的“数据穿越（Data Leakage）”。

---

## * 2. 技术栈 (Tech Stack)
* **核心语言**: `Python 3.8+`
* **医疗影像底座**: 
  * `SimpleITK` (实现三维插值、形态学滤波、全空间矩阵计算)
  * `pydicom` (极速剥离 DICOM 头部脱敏临床元数据)
* **科学计算与特征聚合**: 
  * `pandas` (构建临床大账本、高维度聚合与阵营级联)
  * `numpy` (三维空间距离运算、HU 值截断、安全掩膜填充)

---

## * 3. 项目标准目录结构 (Project Directory Structure)
```text
Lung-Nodule/
│
├── config/
│   └── config.yaml                 # 全局参数与数据集物理路径配置
│
├── data/                           # 数据大本营 (大文件，已被 .gitignore 忽略)
│   ├── raw/                        # LIDC-IDRI (DICOM) / LUNA16 (MHD) 原始目录
│   └── processed/                  # 预处理全流程固化产物
│       ├── preprocessed_cubes/     # 模块一：LUNA16 全肺预处理与分割矩阵 (.npy)
│       ├── all_roi_cubes/          # 模块慢：LIDC 提取的 32x32 结节局部 ROI (.npy)
│       ├── master_data_exploration.csv    # 临床全维度大账本
│       ├── cleaned_nodules_labels.csv     # 高质量黄金结节清单
│       ├── final_dataset_manifest.csv     # 挂载双/多分类标签的资产总账
│       └── final_split_train_manifest.csv # 7:1:2 防泄露切分索引表
│
├── src/                            # 核心闭环源码
│   ├── preprocess.py               # LUNA16 等方性重采样与三维形态学肺实质分割
│   ├── explore_data.py             # LIDC 空间物理聚类与标注一致性清洗
│   ├── extract_roi.py              # 物理坐标对齐、3D ROI 裁切与 HU 截断
│   └── split_dataset.py            # 最大风险锁定、患者级防泄露 7:1:2 发牌
│
└── README.md                       # 本说明文档Lung-Nodule
```

---

## * 4. 工业级流水线运行指南 (Execution Guide)
为确保整个数据流动管道平滑闭环且表头无污染，请严格按照以下一键通关顺序执行：

**模块一**：医学图像全肺标准化预处理验证 (基于 LUNA16)
执行端到端的图像几何重构：涵盖去标识化、等方性插值、基于 sitkBall 的 3D 闭运算肺实质分割，以及空气掩膜的安全填充。
```text
python src/preprocess.py
```

**模块二**：多维特征清洗与 3D ROI 构建 (基于 LIDC-IDRI)
* **第一步**：数据解构、空间聚类与临床大清洗
基于欧氏距离融合多位医生的异构标注，应用核心过滤网。
```text
python src/explore_data.py
```

* **第二步**：3D ROI 物理裁切与多分类特征绑定
根据绝对物理坐标裁切立方体，并行挂载良恶性及多风险标签。
```text
python src/explore_data.py
```

* **第三步**：最大风险锁定与 7:1:2 防泄露切分
以患者 ID 为底层单元执行洗牌，对齐 72% : 10% : 18% 黄金交叉，输出 3D-CNN DataLoader 的标准索引。
```text
python src/extract_roi.py
```

---

## * 5. 快速开始与环境配置 (Quick Start)

# 1. 使用 conda 创建纯净虚拟环境
conda create -n lung_nodule python=3.9 -y
conda activate lung_nodule

# 2. 一键安装高吞吐医学影像处理依赖链
pip install pandas numpy pyyaml pydicom SimpleITK scikit-image scipy

# 3. 校验路径
# 请检查 config/config.yaml 中的 raw_lidc_dir 和 raw_luna_dir 是否指向正确的原始硬盘路径。
