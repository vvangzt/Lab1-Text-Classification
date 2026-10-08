import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# ================= 0. 全局配置与环境锁定 (确保 100% 可复现) =================
SEED = 42
np.random.seed(SEED)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ================= 1. 深度文本清洗 (与逻辑回归严格一致) =================
print(">>> [1/4] 加载数据并执行清洗 (保留英文字母与数字 0-9)...")
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_data.csv'))

def clean_newsgroup_text(text):
    if not isinstance(text, str):
        return ""
    header_pattern = r'^(From|Lines|Reply-To|NNTP-Posting-Host|Organization|Distribution|Keywords|Path|Message-ID|Date|Article-I.D.|Sender|Followup-To):.*$'
    text = re.sub(header_pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\S+@\S+|http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text.lower()).strip()

clean_train_texts = [clean_newsgroup_text(t) for t in train_df['text']]
y_all = train_df['target'].values

# 8:2 分层切分
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    clean_train_texts, y_all,
    test_size=0.2,
    random_state=SEED,
    stratify=y_all
)
print(f"数据划分完毕: 训练集 {len(X_train_raw)} 条, 验证集 {len(X_val_raw)} 条")

# ================= 2. TF-IDF 特征工程 (12000 维) =================
print(">>> [2/4] 构建 TF-IDF 特征工程矩阵...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    max_features=12000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train_raw)
X_val_tfidf = vectorizer.transform(X_val_raw)
print(f"特征矩阵构建完毕: 词表特征维度 = {X_train_tfidf.shape[1]}")

# ================= 3. 训练基准 MLP 模型 =================
print("\n>>> [3/4] 开始训练基准 MLP 神经网络...")
print("网络架构: [输入 12000] -> [隐藏层 100 神经元 (ReLU)] -> [输出 10 类别 (Softmax)]")
print("优化算法: Adam, 初始学习率: 0.001, 最大轮次: 200")
print("-" * 65)

# hidden_layer_sizes=(100,) 表示单层 100 神经元
# verbose=True 会在终端逐轮打印 Loss 下降情况
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=SEED,
    verbose=True
)

start_t = time.time()
mlp.fit(X_train_tfidf, y_train)
cost_time = time.time() - start_t

print("-" * 65)
print(f"MLP 训练完毕! 耗时: {cost_time:.2f} 秒")
print(f"实际迭代轮数 (Epochs): {len(mlp.loss_curve_)}")
print(f"初始损失: {mlp.loss_curve_[0]:.4f} -> 最终收敛损失: {mlp.loss_curve_[-1]:.4f}")

# 计算指标
tr_preds = mlp.predict(X_train_tfidf)
val_preds = mlp.predict(X_val_tfidf)

tr_acc = accuracy_score(y_train, tr_preds)
val_acc = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds, average='macro')
gap = tr_acc - val_acc

print("\n" + "="*65)
print("=== 基准 MLP (Hidden=100) 评估结果 ===")
print(f"训练集准确率 (Train Acc):       {tr_acc:.4f} (100.0%)")
print(f"验证集准确率 (Val Acc):         {val_acc:.4f}")
print(f"验证集宏平均 F1 (Val Macro-F1): {val_f1:.4f}")
print(f"泛化差距 (Train - Val Gap):     {gap:.4f} ({gap*100:.2f}%)")
print("="*65)

# ================= 4. 动态绘制训练损失函数变化图 (Loss Curve) =================
print("\n>>> [4/4] 正在动态绘制损失函数收敛曲线...")
epochs = range(1, len(mlp.loss_curve_) + 1)

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(epochs, mlp.loss_curve_, marker='o', markersize=3.5, 
        color='#1f77b4', linewidth=2.2, label='训练集交叉熵损失 (Training Cross-Entropy Loss)')

# 标注初始点
ax.scatter([1], [mlp.loss_curve_[0]], color='#d62728', s=80, zorder=5)
ax.annotate(f'起始: {mlp.loss_curve_[0]:.3f}', 
            xy=(1, mlp.loss_curve_[0]), xytext=(3, mlp.loss_curve_[0] - 0.15),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.2, headwidth=5),
            fontsize=9.5, weight='bold')

# 标注收敛点
final_ep = len(mlp.loss_curve_)
final_l = mlp.loss_curve_[-1]
ax.scatter([final_ep], [final_l], color='#2ca02c', s=80, zorder=5)
ax.annotate(f'最终收敛: {final_l:.4f}\n(第 {final_ep} 轮停止)', 
            xy=(final_ep, final_l), 
            xytext=(max(1, final_ep - 8), final_l + 0.35),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.2, headwidth=5),
            fontsize=9.5, weight='bold')

ax.set_title("多层感知机 (MLP) 训练损失函数收敛曲线 (Loss Curve Across Epochs)", fontsize=12, pad=12, weight='bold')
ax.set_xlabel("迭代轮次 (Epochs / Iterations)", fontsize=11)
ax.set_ylabel("交叉熵损失 (Loss Value)", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', fontsize=10, frameon=True)

plt.tight_layout()

loss_curve_path = os.path.join(FIGURES_DIR, "mlp_loss_curve.png")
plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✔ 核心损失函数变化图已成功保存至: {loss_curve_path}")