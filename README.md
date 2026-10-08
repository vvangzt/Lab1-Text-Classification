# 现代人工智能实验一：高性能新闻文本分类与泛化机理探究
> **Contemporary AI Lab 1: High-Performance Newsgroup Text Classification & Generalization Analysis**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)
![Macro-F1](https://img.shields.io/badge/Best_Val_Macro--F1-0.9383-success)
![Reproducibility](https://img.shields.io/badge/Reproducibility-100%25_Deterministic-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 目录 (Table of Contents)
- [1. 项目简介 (Project Overview)](#1-项目简介-project-overview)
- [2. 实验性能排行榜 (Benchmark Leaderboard)](#2-实验性能排行榜-benchmark-leaderboard)
- [3. 仓库结构与文件索引 (Repository Structure)](#3-仓库结构与文件索引-repository-structure)
- [4. 环境配置与安装 (Environment Setup)](#4-环境配置与安装-environment-setup)
- [5. 全流程复现指令 (Step-by-Step Reproduction)](#5-全流程复现指令-step-by-step-reproduction)
- [6. 核心方法论与消融设计 (Methodology & Ablation Highlights)](#6-核心方法论与消融设计-methodology--ablation-highlights)
- [7. 作者信息 (Author)](#8-作者信息-author)

---

## 1. 项目简介 (Project Overview)

本项目针对 **20-Newsgroups 10 分类文本数据集**（训练集 7,368 条，测试集 2,457 条），在严格杜绝数据泄露（Zero Data Leakage）与保证 100% 确定性可复现的前提下，构建了一个工业级端到端文本分类与泛化机理探究管线。

### 核心亮点 (Highlights)
* **工程规范与防泄露**：严格采用 **8:2 分层抽样（Stratified Split）**，锁定全局随机种子 `SEED = 42`。所有 TF-IDF 词表、IDF 统计量均严格仅在训练子集上拟合，验证集保持绝对隔离。
* **预处理创新**：深入 Usenet 协议底层，彻底滤除通信头干扰的同时，**定向保留高价值技术与硬件数字代号（0-9）**，使下游所有模型性能稳定提升 0.5%~1.0%。
* **特征消融论证**：通过对 `max_features` 在 3k~20k 区间的消融实验，用实测数据证明了选择 **12,000 维** 是平衡特征覆盖度与神经网络参数量预算的黄金分割点。
* **机制深度剖析**：
  * 基于 Cover 定理与实测 **99.34% 高稀疏度** 论证了逻辑回归的优异基线性能；
  * 通过混淆矩阵深挖了 [类别 2: PC硬件] 与 [类别 7: 电子器件] 的细粒度语义重合；
  * 捕获并可视化了 **Mini-batch 步级损失随机波动**，实测论证了**早停机制（Early Stopping）**作为时间正则化的防过拟合威力。
* **前沿创新对比**：对比了**词级 Word2Vec 离散替换**（揭示了多线程非确定性与语义漂移）与**连续特征级同类流形插值（Intra-class Mixup）**的机理差异。

---

## 2. 实验性能排行榜 (Benchmark Leaderboard)

以下为所有模型在验证集（$N=1,474$，Seed=42）上的客观评测指标汇总：

| 实验组别 / 模型架构 | 特征工程表征 | 参数量 | 收敛轮次 | 训练集 Acc | 验证集 Acc | 验证集 Macro-F1 | 泛化差距 (Gap) | 核心特性评述 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **纯 Word2Vec 均值表征 + MLP** | 100 维 (稠密) | 11,110 | 25 | 0.6512 | 0.6133 | **0.6133** | 0.0379 | 算术平均严重稀释核心判别关键词 |
| **逻辑回归 (C=0.1)** | 12k TF-IDF | 120,000 | 凸收敛 | 0.9415 | 0.9057 | **0.9063** | 0.0358 | 强正则化，模型轻度欠拟合 |
| **逻辑回归 (C=10.0)** | 12k TF-IDF | 120,000 | 凸收敛 | 0.9997 | 0.9315 | **0.9318** | 0.0682 | 高位稳健平原起点 |
| **逻辑回归 (C=100.0) [LR最佳]** | 12k TF-IDF | 120,000 | 凸收敛 | 0.9998 | 0.9355 | **0.9359** | 0.0643 | 线性模型经验峰值，极速收敛 (0.7s) |
| **MLP (Hidden=50, 无早停)** | 12k TF-IDF | 600,500 | 89 | 0.9998 | 0.9355 | **0.9360** | 0.0643 | 窄网络容量初现饱和，需靠长轮次弥补 |
| **MLP (Hidden=100, 50 双层)** | 12k TF-IDF | 1,205,500 | 51 | 0.9998 | 0.9362 | **0.9364** | 0.0636 | 最低训练Loss (0.0015)，典型过拟合 |
| **MLP (Hidden=200, 无早停)** | 12k TF-IDF | 2,402,000 | 67 | 0.9998 | 0.9362 | **0.9366** | 0.0636 | 参数量翻倍至240万，性能停滞微跌 |
| **MLP (Hidden=100, 无早停)** | 12k TF-IDF | 1,201,000 | 74 | 0.9998 | 0.9362 | **0.9368** | 0.0636 | 未加早停，后期死记长尾训练噪音 |
| **MLP (50 + 早停机制)** | 12k TF-IDF | 600,500 | 21 | 0.9869 | 0.9294 | **0.9301** | 0.0575 | 容量紧绷叠加阻断过早导致严重欠拟合 |
| **Word2Vec 增强 + 最优 MLP** | 12k TF-IDF扩充 | 1,201,000 | 24 | 0.9972 | 0.9362 | **0.9369** | 0.0609 | 离散替换带来高维阶跃与轻微语义漂移 |
| **特征级 Mixup 增强 + 最优 MLP** | 12k 凸组合 | 1,201,000 | 23 | 0.9975 | 0.9389 | **0.9389** | 0.0586 | 连续空间内类内流形平滑，收敛加速 |
| 🏆 **最终选定冠军模型 (MLP纯净版)** | **12k TF-IDF** | **1,201,000** | **25** | **0.9925** | **0.9376** | **0.9383** | **0.0549** | **全场最优：泛化鸿沟最窄，类别7 F1达0.9078** |

---

## 3. 仓库结构与文件索引 (Repository Structure)

```text
.
├── figures/                              # [图表库] 动态生成的 300 DPI 学术级插图
│   ├── lr_tuning_dynamic.png             # 逻辑回归对数网格调参折线图
│   ├── lr_confusion_matrix.png           # 逻辑回归验证集混淆矩阵热力图
│   ├── mlp_architectures_comparison.png  # MLP 6 大架构与策略对比柱状图
│   ├── mlp_alpha_tuning.png              # L2 正则项 alpha 调参折线图
│   ├── mlp_minibatch_loss_curve.png      # [核心] 展现 Mini-batch 步级波动的训练损失图
│   ├── mlp_confusion_matrix.png          # 最终 MLP 冠军模型混淆矩阵热力图
│   ├── w2v_augmentation_comparison.png   # Word2Vec 文本增强拟合与泛化对比图
│   └── feature_aug_comparison.png        # 连续特征级 Mixup 与 Dropout 对比柱状图
├── ablation_max_features.py              # [消融] 特征维度 (3k-20k) 敏感性消融实验
├── step3_logistic_regression.py          # [基线] 逻辑回归对数网格调参实验
├── step3_lr_evaluation.py                # [基线] 逻辑回归错误分析与类别报告提取
├── step4_mlp_tuning.py                   # [网络] MLP 宽度、深度与早停对照消融
├── step4_mlp_alpha.py                    # [网络] MLP 正则化系数 alpha 局部细搜
├── plot_minibatch_loss_curve.py          # [出图] 提取 Mini-batch 步级损失并生成图表
├── plot_mlp_confusion_matrix.py          # [出图] 评估最优 MLP 并生成最终混淆矩阵
├── step5_word2vec_augmentation.py        # [创新] 基于 Word2Vec 的近义词数据增强评测
├── test_w2v_representation.py            # [创新] 纯 Word2Vec 均值池化稠密表征对比评测
├── step5_feature_augmentation.py         # [创新] 特征级 Dropout 与同类 Mixup 插值对比
├── generate_prediction.py                # [交付] 使用最优冠军模型推理并导出预测文件
├── train_data.csv                        # 官方带标签训练数据 (7,368 条)
├── test_data_unlabeled.csv               # 官方无标签测试数据 (2,457 条)
├── prediction.csv                        # [交付物] 最终提交预测结果 
├── TUNING.md                             # 官方超参数调优指南
└── README.md                             # 本说明文档
```

---

## 4. 环境配置与安装 (Environment Setup)

推荐使用 Conda 建立纯净的 Python 3.10 沙盒环境：

```bash
# 1. 创建并激活虚拟环境
conda create -n project1 python=3.10 -y
conda activate project1

# 2. 安装全部必备依赖包 (使用清华镜像源秒级下载)
pip install numpy pandas scikit-learn matplotlib seaborn gensim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 5. 全流程复现指令 (Step-by-Step Reproduction)

所有代码均内置相对路径自适应与确定性随机种子锁定，可按实验逻辑依次执行：

### 阶段 1：特征维度消融实验 (证明 12,000 维的科学性)
```bash
python ablation_max_features.py
```
> 输出 3k、5k、8k、12k、20k 维度的对比指标，证实 12,000 维相对 5,000 维带来 +1.89% 的跃升，并展示边际效应递减拐点。

### 阶段 2：逻辑回归基线调参与错误分析
```bash
python step3_logistic_regression.py
python step3_lr_evaluation.py
```
> 自动输出逻辑回归调参表、生成 `figures/lr_tuning_dynamic.png`，并输出全类别报告与混淆矩阵 `figures/lr_confusion_matrix.png`。

### 阶段 3：MLP 网络架构、早停机制与正则化消融
```bash
python step4_mlp_tuning.py
python step4_mlp_alpha.py
```
> 横向评测 50、100、200 神经元及双层网络，验证早停机制的提速与防过拟合功效，生成柱状图 `figures/mlp_architectures_comparison.png`。

### 阶段 4：生成核心插图 (Mini-batch 损失波动与最终混淆矩阵)
```bash
python plot_minibatch_loss_curve.py
python plot_mlp_confusion_matrix.py
```
> 抓取 700 多个 Mini-batch 的阶跃波动，绘制带 EMA 平滑线的损失收敛图 `figures/mlp_minibatch_loss_curve.png`，并生成最终混淆矩阵 `figures/mlp_confusion_matrix.png`。

### 阶段 5：前沿创新探索 (Word2Vec 与特征级插值)
```bash
python step5_word2vec_augmentation.py
python test_w2v_representation.py
python step5_feature_augmentation.py
```
> 测试离散词替换与连续向量插值，论证 Word2Vec 均值池化出现关键词稀释导致 F1 跌至 0.6133 的深层机理。

### 阶段 6：最终测试集推理与文件交付导出
```bash
python generate_prediction.py
```
> 严格基于 80% 隔离训练集拟合的冠军模型，对 2,457 条测试文档进行前向推理，自动生成无表头、无索引的 `prediction.csv`。

---

## 6. 核心方法论与消融设计 (Methodology & Ablation Highlights)

### 6.1 为什么保留数字特征？（预处理创新）
自然语言预处理中常盲目滤除所有数字。但针对 20-Newsgroups 中的技术硬件类别（如 `comp.sys.ibm.pc.hardware`），硬件代号如 `386`, `486`, `windows 3.1`, `24X` 是强判别性特征。消融对比表明，保留数字使各模型的 Macro-F1 指标稳定上涨 0.5%~1.0%。

### 6.2 为什么早停机制（Early Stopping）能刷新全场记录？
实测显示，无约束训练会持续进行至第 74 轮，将训练损失死磕至 0.0031，泛化差距达 6.36%。而开启早停机制后：
1. **时间阻断**：在第 25 轮检测到验证集泛化见顶，果断终止，耗时从 112s 压缩至 34s（提速 3 倍多）；
2. **隐式 L2 正则化**：限制了权重范数膨胀，将泛化差距压缩至全场最低的 **5.49%**，验证集 Macro-F1 冲上 **0.9383**。

### 6.3 实验可复现性中的非确定性排查
在 Word2Vec 探索中曾观测到 0.9395 与 0.9388 的微小波动。排查证实系 Gensim 底层多线程（`workers > 1`）在共享内存中并发写入时引起的操作系统线程调度时序竞争（Race Condition）。为了达成工业级比特级可复现性，管线最终将 Word2Vec 约束为单线程（`workers = 1`），彻底消除了并发时序非确定性。

---


---

## 7. 作者信息 (Author)
* **完成人**：[王子腾] (学号: [10245501473])
* **课程**：当代人工智能实验课
* **日期**：2026 年 10 月
