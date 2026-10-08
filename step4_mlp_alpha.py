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

# ================= 0. 全局配置与环境锁定 =================
SEED = 42
np.random.seed(SEED)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ================= 1. 数据加载与清洗 =================
print(">>> [1/3] 加载数据并清洗 (保留英文字母与数字 0-9)...")
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

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    clean_train_texts, y_all,
    test_size=0.2,
    random_state=SEED,
    stratify=y_all
)

# ================= 2. TF-IDF 特征工程 (12000 维) =================
print(">>> [2/3] 构建 TF-IDF 特征矩阵...")
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

# ================= 3. 专注测试 alpha (锁定 100 神经元 + 早停) =================
print("\n>>> [3/3] 开始测试 L2 正则化参数 alpha (锁定: Hidden=100 + 早停)...")
alpha_candidates = [0.0001, 0.001, 0.01, 0.1]
results = []

for a in alpha_candidates:
    start_t = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        alpha=a,              # 遍历 L2 权重衰减
        early_stopping=True,  # 开启早停
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=SEED
    )
    mlp.fit(X_train_tfidf, y_train)
    cost = time.time() - start_t
    
    tr_acc = accuracy_score(y_train, mlp.predict(X_train_tfidf))
    v_preds = mlp.predict(X_val_tfidf)
    v_acc = accuracy_score(y_val, v_preds)
    v_f1 = f1_score(y_val, v_preds, average='macro')
    gap = tr_acc - v_acc
    
    results.append({
        "alpha (L2惩罚)": a,
        "刹车轮次": len(mlp.loss_curve_),
        "最终Loss": mlp.loss_curve_[-1],
        "训练集Acc": tr_acc,
        "验证集Acc": v_acc,
        "验证集Macro-F1": v_f1,
        "泛化差距(Gap)": gap,
        "耗时(秒)": cost
    })
    print(f"  • alpha={a:<6} 训练完成 (轮次: {len(mlp.loss_curve_)}, 验证集F1: {v_f1:.4f}, 耗时: {cost:.1f}s)")

res_df = pd.DataFrame(results)

print("\n" + "="*90)
print("=== MLP 正则化参数 alpha (L2项) 客观实验结果汇总表 ===")
print(res_df.to_string(index=False, justify='center', float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
print("="*90)

# ================= 4. 动态绘制 alpha 调参折线图 =================
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(res_df['alpha (L2惩罚)'], res_df['训练集Acc'], marker='o', linewidth=2.2, label='训练集准确率 (Train Acc)', color='#d62728')
ax.plot(res_df['alpha (L2惩罚)'], res_df['验证集Acc'], marker='s', linewidth=2.0, label='验证集准确率 (Val Acc)', color='#1f77b4')
ax.plot(res_df['alpha (L2惩罚)'], res_df['验证集Macro-F1'], marker='^', linewidth=2.0, linestyle='--', label='验证集宏平均F1 (Val Macro-F1)', color='#2ca02c')

ax.set_xscale('log')
ax.set_title("MLP 正则化参数 alpha 对模型性能的影响 (Hidden=100 + 早停)", fontsize=12, pad=10, weight='bold')
ax.set_xlabel("L2 正则化系数 alpha (对数刻度)", fontsize=11)
ax.set_ylabel("指标得分 (Score)", fontsize=11)
ax.set_ylim(0.85, 1.02)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower left', fontsize=10, frameon=True)

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "mlp_alpha_tuning.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✔ 调参折线图已保存至: {fig_path}")
print(">>> 客观数据全部输出完毕，请查看结果表。")