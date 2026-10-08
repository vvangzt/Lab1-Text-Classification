import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ================= 0. 全局配置与环境锁定 (确保 100% 可复现) =================
SEED = 42
np.random.seed(SEED)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 智能兼容目录层级
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = CURRENT_DIR if not CURRENT_DIR.endswith("figures") else os.path.dirname(CURRENT_DIR)
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ================= 1. 数据加载与清洗 (保留数字 0-9) =================
print(">>> [1/4] 加载训练集并执行清洗...")
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
print(">>> [2/4] 构建高质量 TF-IDF 矩阵 (12000 维)...")
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

# ================= 3. 训练最终选定的最优 MLP 冠军模型 =================
print(">>> [3/4] 训练最终 MLP 模型: Hidden=100, 早停机制, alpha=0.0010...")
final_mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    alpha=0.0010,
    early_stopping=True,
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=SEED
)

final_mlp.fit(X_train_tfidf, y_train)

val_preds = final_mlp.predict(X_val_tfidf)
val_acc = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds, average='macro')

print(f"\n✔ 评估完成 (轮次: {len(final_mlp.loss_curve_)} 轮): 验证集 Acc = {val_acc:.4f}, Macro-F1 = {val_f1:.4f}")

# ================= 4. 输出完整分类报告 =================
print("\n" + "="*70)
print("=== 最优 MLP (Hidden=100, 早停, alpha=0.0010) 各类别详细分类报告 ===")
print("="*70)
print(classification_report(y_val, val_preds, target_names=[f"类别 {i}" for i in range(10)], digits=4))

# ================= 5. 绘制并保存混淆矩阵热力图 =================
print(">>> [4/4] 正在绘制并保存混淆矩阵热力图...")
cm = confusion_matrix(y_val, val_preds)

fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=True,
    xticklabels=[f"C{i}" for i in range(10)],
    yticklabels=[f"C{i}" for i in range(10)],
    ax=ax, linewidths=0.5, linecolor='lightgray'
)

ax.set_title(f"MLP (Hidden=100, 早停) 验证集混淆矩阵热力图 (Acc: {val_acc*100:.2f}%)", 
             fontsize=12, pad=12, weight='bold')
ax.set_xlabel("模型预测标签 (Predicted Label)", fontsize=11)
ax.set_ylabel("真实数据标签 (True Label)", fontsize=11)
plt.tight_layout()

cm_path = os.path.join(FIGURES_DIR, "mlp_confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

# ================= 6. 自动错误归因：找出最容易混淆的类别 Top-3 =================
print("="*70)
print("=== 错误分析 (Error Analysis): MLP 最容易混淆的类别 Top-3 ===")
confusion_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i][j] > 0:
            confusion_pairs.append((i, j, cm[i][j]))

confusion_pairs.sort(key=lambda x: x[2], reverse=True)

for rank, (true_cls, pred_cls, count) in enumerate(confusion_pairs[:3], 1):
    print(f"Top {rank}: 真实为 [类别 {true_cls}]，却被误判为 [类别 {pred_cls}]，共发生 {count} 次")
print("="*70)
print(f"✔ MLP 混淆矩阵已生成并保存至: {cm_path}")