import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# ================= 0. 全局配置与环境锁定 =================
SEED = 42
np.random.seed(SEED)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = CURRENT_DIR if not CURRENT_DIR.endswith("figures") else os.path.dirname(CURRENT_DIR)
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ================= 1. 数据加载与清洗 =================
print(">>> [1/3] 加载数据并清洗 (保留数字)...")
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

# ================= 2. TF-IDF 特征工程 =================
print(">>> [2/3] 构建 TF-IDF 特征矩阵 (12000 维)...")
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

# ================= 3. 训练最终冠军模型并提取真实 Loss 序列 =================
print(">>> [3/3] 训练最终冠军模型 (Hidden=100, 早停, alpha=0.0010)...")
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

# 提取关键过程数据
loss_history = final_mlp.loss_curve_
val_score_history = final_mlp.validation_scores_
epochs = range(1, len(loss_history) + 1)

print(f"训练完成: 实际收敛轮数 = {len(loss_history)} 轮, 最终训练损失 = {loss_history[-1]:.4f}")

# ================= 4. 绘制高密度学术级双轴收敛曲线图 =================
fig, ax1 = plt.subplots(figsize=(7.5, 4.2))

# 左 Y 轴：绘制训练交叉熵损失 (红色)
color_loss = '#d62728'
ax1.set_xlabel('迭代轮次 (Epochs / Iterations)', fontsize=10.5)
ax1.set_ylabel('训练集交叉熵损失 (Loss)', color=color_loss, fontsize=10.5, weight='bold')
line1 = ax1.plot(epochs, loss_history, color=color_loss, linewidth=2.2, marker='o', markersize=3.5, label='训练交叉熵损失 (Loss)')
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.set_ylim(-0.05, 2.35)
ax1.grid(True, linestyle='--', alpha=0.4)

# 标注起始损失
ax1.annotate(f'初始: {loss_history[0]:.2f}', xy=(1, loss_history[0]), xytext=(3, loss_history[0] - 0.2),
             arrowprops=dict(facecolor=color_loss, shrink=0.05, width=1, headwidth=4),
             fontsize=8.5, color=color_loss, weight='bold')

# 标注终止损失与轮次
final_ep = len(loss_history)
ax1.annotate(f'早停终止 (第 {final_ep} 轮)\nLoss={loss_history[-1]:.4f}', 
             xy=(final_ep, loss_history[-1]), xytext=(final_ep - 11, loss_history[-1] + 0.45),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.2, headwidth=5),
             fontsize=8.5, weight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="#fff9c4", ec="#fbc02d", alpha=0.9))

# 右 Y 轴：绘制早停监视的验证集精度 (蓝色)
ax2 = ax1.twinx()
color_score = '#1f77b4'
ax2.set_ylabel('早停监测验证集精度 (Accuracy)', color=color_score, fontsize=10.5, weight='bold')
line2 = ax2.plot(epochs, val_score_history, color=color_score, linewidth=2.0, linestyle='--', marker='s', markersize=3.5, label='验证集精度 (Score)')
ax2.tick_params(axis='y', labelcolor=color_score)
ax2.set_ylim(0.70, 0.98)

# 标出最佳点 
best_val_epoch = np.argmax(val_score_history) + 1
best_val_score = np.max(val_score_history)
ax2.scatter([best_val_epoch], [best_val_score], color='gold', s=100, edgecolors='black', zorder=5)
ax2.annotate(f'泛化极值点 (第 {best_val_epoch} 轮)', xy=(best_val_epoch, best_val_score), xytext=(best_val_epoch - 9, best_val_score - 0.06),
             arrowprops=dict(facecolor=color_score, shrink=0.05, width=1, headwidth=4),
             fontsize=8.5, color=color_score, weight='bold')

# 合并图例放在图表中央偏上位置
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=9, frameon=True)

plt.title("MLP 最终模型训练损失与早停泛化曲线变化图 (Hidden=100, α=0.001)", fontsize=11.5, pad=10, weight='bold')
plt.tight_layout()

save_path = os.path.join(FIGURES_DIR, "mlp_final_loss_curve.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✔ 最终损失函数变化图已生成至: {save_path}")
print(">>> 核心图表已全部齐备！")