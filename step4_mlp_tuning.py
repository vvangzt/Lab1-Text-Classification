import os
import re
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ================= 0. 全局配置与环境锁定 =================
SEED = 42
np.random.seed(SEED)

matplotlib.rcParams['font.sans-serif'] = [
    'SimHei',
    'Microsoft YaHei',
    'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ================= 1. 数据加载与清洗 (保留数字) =================
print('>>> [1/3] 加载数据并执行清洗 (保留英文字母与数字 0-9)...')
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_data.csv'))


def clean_newsgroup_text(text):
  if not isinstance(text, str):
    return ''
  header_pattern = r'^(From|Lines|Reply-To|NNTP-Posting-Host|Organization|Distribution|Keywords|Path|Message-ID|Date|Article-I.D.|Sender|Followup-To):.*$'
  text = re.sub(header_pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
  text = re.sub(r'\S+@\S+|http\S+|www\S+', '', text)
  text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
  return re.sub(r'\s+', ' ', text.lower()).strip()


clean_train_texts = [clean_newsgroup_text(t) for t in train_df['text']]
y_all = train_df['target'].values

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    clean_train_texts, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

# ================= 2. TF-IDF 特征工程 (12000 维) =================
print('>>> [2/3] 构建 TF-IDF 特征工程矩阵...')
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    max_features=12000,
    sublinear_tf=True,
)

X_train_tfidf = vectorizer.fit_transform(X_train_raw)
X_val_tfidf = vectorizer.transform(X_val_raw)

# ================= 3. 客观测试不同网络结构与早停机制 =================
print('\n>>> [3/3] 开始训练不同网络架构的 MLP 并记录表现...')

architectures = [
    {'name': 'MLP (Hidden=50)', 'hidden': (50,), 'early_stop': False},
    {'name': 'MLP (Hidden=100)', 'hidden': (100,), 'early_stop': False},
    {'name': 'MLP (Hidden=200)', 'hidden': (200,), 'early_stop': False},
    {'name': 'MLP (Hidden=100, 50)', 'hidden': (100, 50), 'early_stop': False},
    {
        'name': 'MLP (50 + 早停机制)',
        'hidden': (50,),
        'early_stop': True,
    },  # <-- 新增：验证你的深度假设！
    {'name': 'MLP (100 + 早停机制)', 'hidden': (100,), 'early_stop': True},
]

results = []
trained_models = {}

for arch in architectures:
  start_t = time.time()

  mlp = MLPClassifier(
      hidden_layer_sizes=arch['hidden'],
      activation='relu',
      solver='adam',
      max_iter=200,
      early_stopping=arch['early_stop'],
      random_state=SEED,
  )

  mlp.fit(X_train_tfidf, y_train)
  cost = time.time() - start_t

  tr_acc = accuracy_score(y_train, mlp.predict(X_train_tfidf))
  v_preds = mlp.predict(X_val_tfidf)
  v_acc = accuracy_score(y_val, v_preds)
  v_f1 = f1_score(y_val, v_preds, average='macro')
  gap = tr_acc - v_acc
  final_loss = mlp.loss_curve_[-1]
  stopped_epoch = len(mlp.loss_curve_)

  results.append({
      '网络结构': arch['name'],
      '参数规模(估算)': f'{sum([w.size for w in mlp.coefs_]):,}',
      '收敛轮次': stopped_epoch,
      '最终Loss': final_loss,
      '训练集Acc': tr_acc,
      '验证集Acc': v_acc,
      '验证集Macro-F1': v_f1,
      '泛化差距(Gap)': gap,
      '耗时(秒)': cost,
  })
  trained_models[arch['name']] = mlp
  print(
      f"  • {arch['name']:<22} 训练完成 (耗时: {cost:.1f}s, 轮次:"
      f' {stopped_epoch}, 验证集F1: {v_f1:.4f})'
  )

res_df = pd.DataFrame(results)

print('\n' + '=' * 100)
print('=== 多层感知机 (MLP) 全架构与早停对照实验结果汇总表 ===')
print(
    res_df.to_string(
        index=False,
        justify='center',
        float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x,
    )
)
print('=' * 100)

# ================= 4. 动态更新对比柱状图 =================
fig, ax = plt.subplots(figsize=(11, 5.8))
x_indices = np.arange(len(res_df))
bar_width = 0.35

rects1 = ax.bar(
    x_indices - bar_width / 2,
    res_df['训练集Acc'],
    bar_width,
    label='训练集准确率 (Train Acc)',
    color='#d62728',
    alpha=0.85,
)
rects2 = ax.bar(
    x_indices + bar_width / 2,
    res_df['验证集Macro-F1'],
    bar_width,
    label='验证集宏平均F1 (Val Macro-F1)',
    color='#1f77b4',
    alpha=0.85,
)

ax.set_title(
    '不同 MLP 网络架构与早停策略的性能对比 (含 50 与 100 早停对照)',
    fontsize=12,
    pad=12,
    weight='bold',
)
ax.set_ylabel('指标得分 (Score)', fontsize=11)
ax.set_xticks(x_indices)
ax.set_xticklabels(res_df['网络结构'], fontsize=9.5, rotation=15)
ax.set_ylim(0.85, 1.03)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for rect in rects1:
  h = rect.get_height()
  ax.annotate(
      f'{h:.3f}',
      xy=(rect.get_x() + rect.get_width() / 2, h),
      xytext=(0, 2),
      textcoords='offset points',
      ha='center',
      va='bottom',
      fontsize=8.5,
  )

for rect in rects2:
  h = rect.get_height()
  ax.annotate(
      f'{h:.4f}',
      xy=(rect.get_x() + rect.get_width() / 2, h),
      xytext=(0, 2),
      textcoords='offset points',
      ha='center',
      va='bottom',
      fontsize=9,
      weight='bold',
  )

ax.legend(loc='lower right', fontsize=10, frameon=True)
plt.tight_layout()

compare_fig_path = os.path.join(FIGURES_DIR, 'mlp_architectures_comparison.png')
plt.savefig(compare_fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'\n✔ 全量对照柱状图已更新并保存至: {compare_fig_path}')