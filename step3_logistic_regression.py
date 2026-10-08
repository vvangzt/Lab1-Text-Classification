import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 补全缺失的 sklearn 模块导入
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ================= 0. 全局配置与环境锁定 (确保 100% 可复现) =================
SEED = 42
np.random.seed(SEED)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ================= 1. 深度文本清洗 (保留英文字母与数字) =================
print(">>> [1/3] 正在加载数据并执行清洗 (保留英文字母与数字 0-9)...")
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_data.csv'))

def clean_newsgroup_text(text):
    if not isinstance(text, str):
        return ""
    # 彻底清除常见的通讯头 (From, Lines, Path, Keywords, Date 等)
    header_pattern = r'^(From|Lines|Reply-To|NNTP-Posting-Host|Organization|Distribution|Keywords|Path|Message-ID|Date|Article-I.D.|Sender|Followup-To):.*$'
    text = re.sub(header_pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    # 清理网址与邮箱
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    # 保留英文字母和数字 (a-zA-Z0-9)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # 统一小写并压缩连续空格
    text = re.sub(r'\s+', ' ', text.lower()).strip()
    return text

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

# ================= 2. TF-IDF 特征工程 =================
print(">>> [2/3] 构建 TF-IDF 特征工程矩阵 (min_df=2, max_df=0.9, max_features=12000)...")
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

# ================= 3. 逻辑回归：客观评测各候选参数 =================
print(">>> [3/3] 正在对各候选参数进行训练与评估...")
c_candidates = [0.1, 1.0, 10.0, 100.0,1000.0]
results = []

for c in c_candidates:
    start_t = time.time()
    lr = LogisticRegression(C=c, max_iter=2000, random_state=SEED)
    lr.fit(X_train_tfidf, y_train)
    cost = time.time() - start_t
    
    tr_acc = accuracy_score(y_train, lr.predict(X_train_tfidf))
    v_preds = lr.predict(X_val_tfidf)
    v_acc = accuracy_score(y_val, v_preds)
    v_f1 = f1_score(y_val, v_preds, average='macro')
    gap = tr_acc - v_acc  # 泛化差距 (衡量过拟合程度)
    
    results.append({
        "C 参数": c,
        "训练集 Acc": tr_acc,
        "验证集 Acc": v_acc,
        "验证集 Macro-F1": v_f1,
        "泛化差距 (Train-Val)": gap,
        "训练耗时(秒)": cost
    })

res_df = pd.DataFrame(results)

print("\n" + "="*80)
print("=== 逻辑回归各参数真实实验结果汇总表 ===")
print(res_df.to_string(index=False, justify='center', float_format=lambda x: f"{x:.4f}"))
print("="*80)

# ================= 4. 动态绘制调参折线图 (完全基于真实实验数据) =================
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(res_df['C 参数'], res_df['训练集 Acc'], marker='o', linewidth=2.2, label='训练集准确率 (Train Acc)', color='#d62728')
ax.plot(res_df['C 参数'], res_df['验证集 Acc'], marker='s', linewidth=2.0, label='验证集准确率 (Val Acc)', color='#1f77b4')
ax.plot(res_df['C 参数'], res_df['验证集 Macro-F1'], marker='^', linewidth=2.0, linestyle='--', label='验证集宏平均F1 (Val Macro-F1)', color='#2ca02c')

ax.set_xscale('log')
ax.set_title("逻辑回归超参数 C 对模型性能的影响 (Seed=42)", fontsize=12, pad=10, weight='bold')
ax.set_xlabel("超参数 C (对数刻度, C = 1 / λ)", fontsize=11)
ax.set_ylabel("指标得分 (Score)", fontsize=11)
ax.set_ylim(0.85, 1.02)
ax.grid(True, linestyle='--', alpha=0.5)

ax.legend(loc='lower right', fontsize=10, frameon=True)
plt.tight_layout()

dynamic_fig_path = os.path.join(FIGURES_DIR, "lr_tuning_dynamic.png")
plt.savefig(dynamic_fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✔ 动态折线图已生成至: {dynamic_fig_path}")
print(">>> 所有参数结果已客观呈现，未进行任何预设立场的选择。")