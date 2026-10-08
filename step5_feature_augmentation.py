import os
import re
import random
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# ================= 0. 全局配置与环境锁定 (确保 100% 可复现) =================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ================= 1. 数据加载与清洗 =================
print(">>> [1/4] 加载数据并清洗 (保留英文字母与数字 0-9)...")
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

# 严格 8:2 分层切分
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    clean_train_texts, y_all,
    test_size=0.2,
    random_state=SEED,
    stratify=y_all
)

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
X_val_tfidf   = vectorizer.transform(X_val_raw)

# ================= 3. 构建特征级数据增强算法 =================
print(">>> [3/4] 正在构造特征级增强数据 (Dropout 与 Intra-class Mixup)...")

# --- 算法 A: 特征 Dropout (随机遮盖 15% 的非零特征) ---
def apply_feature_dropout(X_sparse, drop_rate=0.15):
    X_dropped = X_sparse.copy().tolil()
    for i in range(X_dropped.shape[0]):
        non_zeros = X_dropped.rows[i]
        if len(non_zeros) > 4:
            # 随机挑选 15% 的非零索引强制置 0
            drop_count = int(len(non_zeros) * drop_rate)
            to_drop = random.sample(non_zeros, drop_count)
            for col in to_drop:
                X_dropped[i, col] = 0.0
    return normalize(X_dropped.tocsr())

# 仅对 50% 的样本应用特征 Dropout
sample_indices = random.sample(range(X_train_tfidf.shape[0]), int(X_train_tfidf.shape[0] * 0.5))
X_train_dropped = apply_feature_dropout(X_train_tfidf[sample_indices], drop_rate=0.15)
y_train_dropped = y_train[sample_indices]

# 拼接为 Dropout 扩充集
X_train_dropout_comb = sp.vstack([X_train_tfidf, X_train_dropped])
y_train_dropout_comb = np.concatenate([y_train, y_train_dropped])

# --- 算法 B: 同类别流形插值 (Intra-class Mixup) ---
def apply_intraclass_mixup(X_sparse, y, mix_ratio=0.5):
    synthetic_rows = []
    synthetic_y = []
    
    for c in range(10):
        class_indices = np.where(y == c)[0]
        n_mix = int(len(class_indices) * mix_ratio)
        for _ in range(n_mix):
            idx1, idx2 = random.sample(list(class_indices), 2)
            lam = random.uniform(0.65, 0.85)  # 混合系数
            mixed_vec = lam * X_sparse[idx1] + (1.0 - lam) * X_sparse[idx2]
            synthetic_rows.append(mixed_vec)
            synthetic_y.append(c)
            
    X_synthetic = sp.vstack(synthetic_rows)
    return normalize(X_synthetic), np.array(synthetic_y)

X_train_mixup_synth, y_train_mixup_synth = apply_intraclass_mixup(X_train_tfidf, y_train, mix_ratio=0.5)

# 拼接为 Mixup 扩充集
X_train_mixup_comb = sp.vstack([X_train_tfidf, X_train_mixup_synth])
y_train_mixup_comb = np.concatenate([y_train, y_train_mixup_synth])

print(f"  • 基准训练样本: {X_train_tfidf.shape[0]} 条")
print(f"  • 特征 Dropout 扩充后: {X_train_dropout_comb.shape[0]} 条")
print(f"  • 同类 Mixup 扩充后: {X_train_mixup_comb.shape[0]} 条")

# ================= 4. 对照训练与客观评测 =================
print("\n>>> [4/4] 正在统一训练最强 MLP 冠军架构并进行客观对比...")

champion_params = {
    "hidden_layer_sizes": (100,),
    "alpha": 0.0010,
    "early_stopping": True,
    "activation": "relu",
    "solver": "adam",
    "max_iter": 200,
    "random_state": SEED
}

experiments = [
    {"name": "未增强基准组 (Original)",       "X": X_train_tfidf,        "y": y_train},
    {"name": "特征 Dropout 增强 (Feature Drop)", "X": X_train_dropout_comb, "y": y_train_dropout_comb},
    {"name": "同类流形插值 (Intra-class Mixup)", "X": X_train_mixup_comb,   "y": y_train_mixup_comb}
]

results = []

for exp in experiments:
    start_t = time.time()
    mlp = MLPClassifier(**champion_params)
    mlp.fit(exp["X"], exp["y"])
    cost = time.time() - start_t
    
    tr_acc = accuracy_score(exp["y"], mlp.predict(exp["X"]))
    val_preds = mlp.predict(X_val_tfidf)
    v_acc = accuracy_score(y_val, val_preds)
    v_f1  = f1_score(y_val, val_preds, average='macro')
    gap   = tr_acc - v_acc
    stopped_ep = len(mlp.loss_curve_)
    final_loss = mlp.loss_curve_[-1]
    
    results.append({
        "实验组别": exp["name"],
        "样本总量": exp["X"].shape[0],
        "收敛轮次": stopped_ep,
        "最终Loss": final_loss,
        "训练集Acc": tr_acc,
        "验证集Acc": v_acc,
        "验证集Macro-F1": v_f1,
        "泛化差距(Gap)": gap,
        "耗时(秒)": cost
    })
    print(f"  • {exp['name']} 完成 (轮次: {stopped_ep}, Val F1: {v_f1:.4f}, 耗时: {cost:.1f}s)")

res_df = pd.DataFrame(results)

print("\n" + "="*105)
print("=== 特征级数据增强 (Dropout vs. Mixup) 客观实验结果对比表 ===")
print(res_df.to_string(index=False, justify='center', float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
print("="*105)

# ================= 5. 动态保存对比柱状图 =================
fig, ax = plt.subplots(figsize=(9, 5.2))
x_idx = np.arange(len(res_df))
width = 0.35

rects1 = ax.bar(x_idx - width/2, res_df['训练集Acc'], width, label='训练集准确率 (Train Acc)', color='#d62728', alpha=0.85)
rects2 = ax.bar(x_idx + width/2, res_df['验证集Macro-F1'], width, label='验证集宏平均F1 (Val Macro-F1)', color='#2ca02c', alpha=0.85)

ax.set_ylim(0.90, 1.02)
ax.set_title("特征级数据增强 (Dropout 与 Mixup) 对 MLP 拟合与泛化的影响", fontsize=12, pad=12, weight='bold')
ax.set_ylabel("指标得分 (Score)", fontsize=11)
ax.set_xticks(x_idx)
ax.set_xticklabels(res_df['实验组别'], fontsize=9.5)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                textcoords="offset points", ha='center', va='bottom', fontsize=9)

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                textcoords="offset points", ha='center', va='bottom', fontsize=9.5, weight='bold')

ax.legend(loc='lower right', fontsize=10, frameon=True)
plt.tight_layout()

save_path = os.path.join(FIGURES_DIR, "feature_aug_comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✔ 特征级增强对比柱状图已保存至: {save_path}")
print(">>> 客观数据全部呈现完毕！")