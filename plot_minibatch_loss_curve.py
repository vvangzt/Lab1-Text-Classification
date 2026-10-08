import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

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
print(">>> [1/4] 加载并清洗数据 (保留英文字母与数字 0-9)...")
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

# 严格 8:2 分层切分出内部早停监视集
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

# ================= 3. 逐 Batch 训练并实时记录 Mini-batch 波动 =================
print(">>> [3/4] 启动 Mini-batch 迭代器，实时抓取步级损失震荡...")

BATCH_SIZE = 200
N_SAMPLES = X_train_tfidf.shape[0]
CLASSES = np.unique(y_train)

mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    alpha=0.0010,
    activation='relu',
    solver='adam',
    random_state=SEED
)

step_losses = []
epoch_val_scores = []
best_val_score = -1.0
patience_counter = 0
PATIENCE = 10
MAX_EPOCHS = 40

# 记录早停退出的总步数和轮数
stopped_epoch = MAX_EPOCHS

for epoch in range(1, MAX_EPOCHS + 1):
    # 每轮开始前对训练集执行一次随机洗牌 (Shuffle)
    perm = np.random.permutation(N_SAMPLES)
    X_shuffled = X_train_tfidf[perm]
    y_shuffled = y_train[perm]
    
    # 按 Mini-batch 切片迭代
    for i in range(0, N_SAMPLES, BATCH_SIZE):
        X_batch = X_shuffled[i : i + BATCH_SIZE]
        y_batch = y_shuffled[i : i + BATCH_SIZE]
        
        # 单步前向与反向传播
        mlp.partial_fit(X_batch, y_batch, classes=CLASSES)
        
        # 实时计算当前 Mini-batch 上的真实交叉熵损失
        probs = mlp.predict_proba(X_batch)
        batch_loss = log_loss(y_batch, probs, labels=CLASSES)
        step_losses.append(batch_loss)
        
    # 本轮结束，评估验证集以模拟早停检测
    val_acc = mlp.score(X_val_tfidf, y_val)
    epoch_val_scores.append(val_acc)
    
    if val_acc > best_val_score + 1e-4:
        best_val_score = val_acc
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= PATIENCE:
        stopped_epoch = epoch
        print(f"  • 早停触发: 验证集在第 {epoch - PATIENCE} 轮达到最优峰值，连续 {PATIENCE} 轮未提升，于第 {epoch} 轮终止！")
        break

print(f"训练结束: 共执行 {len(step_losses)} 个 Mini-batch Steps (共 {stopped_epoch} 轮)")

# ================= 4. 计算平滑趋势线 (EMA: 指数移动平均) =================
smooth_losses = []
ema = step_losses[0]
smoothing_factor = 0.05  # 平滑系数
for l in step_losses:
    ema = ema * (1 - smoothing_factor) + l * smoothing_factor
    smooth_losses.append(ema)

# ================= 5. 绘制带高频波动的 Mini-batch 损失曲线 =================
print(">>> [4/4] 正在绘制学术级 Mini-batch 损失波动曲线图...")

fig, ax = plt.subplots(figsize=(8.5, 4.8))
steps = range(1, len(step_losses) + 1)

# 1. 绘制底层的高频波动真实数据 (浅红细线，展现小批量噪声)
ax.plot(steps, step_losses, color='#ff9999', alpha=0.55, linewidth=0.9, 
        label='Mini-batch 实时瞬时损失 (Batch-level Loss, Raw)')

# 2. 绘制前景平滑趋势线 (深红粗实线，展示宏观下敛)
ax.plot(steps, smooth_losses, color='#d62728', linewidth=2.3, 
        label='平滑损失趋势 (Smoothed Trend, EMA)')

# 标注初始高点
ax.annotate(f'起始损失: {step_losses[0]:.2f}', 
            xy=(1, step_losses[0]), xytext=(35, step_losses[0] - 0.2),
            arrowprops=dict(facecolor='#d62728', shrink=0.05, width=1, headwidth=4),
            fontsize=9, weight='bold', color='#d62728')

# 标注早停触发停止点
final_step = len(step_losses)
final_smooth_loss = smooth_losses[-1]
ax.scatter([final_step], [step_losses[-1]], color='black', s=60, zorder=5)
ax.annotate(f'早停终止 (第 {stopped_epoch} 轮 / {final_step} 步)\nLoss ≈ {final_smooth_loss:.4f}', 
            xy=(final_step, step_losses[-1]), 
            xytext=(final_step - 240, step_losses[-1] + 0.5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.2, headwidth=5),
            fontsize=8.5, weight='bold', 
            bbox=dict(boxstyle="round,pad=0.25", fc="#fff9c4", ec="#fbc02d", alpha=0.9))

# 细节排版
ax.set_title("MLP 训练损失函数变化图 (展现 Mini-batch 随机波动与收敛趋势)", fontsize=12, pad=12, weight='bold')
ax.set_xlabel("训练迭代步数 (Training Steps / Mini-batches, Batch Size = 200)", fontsize=10.5)
ax.set_ylabel("交叉熵损失 (Cross-Entropy Loss)", fontsize=10.5)
ax.set_ylim(-0.05, max(step_losses[:10]) + 0.3)
ax.grid(True, linestyle='--', alpha=0.45)
ax.legend(loc='upper right', fontsize=9.5, frameon=True)

plt.tight_layout()

save_path = os.path.join(FIGURES_DIR, "mlp_minibatch_loss_curve.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✔ 展现 Mini-batch 真实波动的损失曲线图已生成至: {save_path}")