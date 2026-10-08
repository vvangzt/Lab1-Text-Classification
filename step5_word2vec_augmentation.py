import os
import random
import re
import time
from gensim.models import Word2Vec
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ================= 0. 全局配置与环境锁定 (确保 100% 可复现) =================
SEED = 42
random.seed(SEED)
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

# ================= 1. 数据加载与清洗 =================
print('>>> [1/5] 加载数据并清洗 (保留英文字母与数字 0-9)...')
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

# 严格 8:2 分层切分
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    clean_train_texts, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)
print(f'原始划分: 训练集 {len(X_train_raw)} 条, 验证集 {len(X_val_raw)} 条')

# ================= 2. 训练领域内 Word2Vec 词向量空间 =================
print(
    '>>> [2/5] 仅在训练集上训练 Word2Vec 语义空间 (绝无验证集数据泄露)...'
)
tokenized_train = [text.split() for text in X_train_raw]

w2v_model = Word2Vec(
    sentences=tokenized_train,
    vector_size=100,
    window=5,
    min_count=3,
    workers=1,
    seed=SEED,
)
print(f'Word2Vec 词表建立完毕，包含有效语义词汇: {len(w2v_model.wv)} 个')

# 抽样展示近义词
sample_words = ['computer', 'space', 'game', 'board']
print('\n--- 抽样查看 Word2Vec 在本语料中学到的空间近义词 ---')
for sw in sample_words:
  if sw in w2v_model.wv:
    sim_words = [w for w, _ in w2v_model.wv.most_similar(sw, topn=3)]
    print(f"  • '{sw}' 的语义近邻词: {sim_words}")
print('-' * 55)

# ================= 3. 执行 Word2Vec 语义近义词替换数据增强 =================
print('\n>>> [3/5] 执行基于 Word2Vec 的同义词替换数据增强...')


def w2v_augment_text(text, w2v_wv, replace_prob=0.15):
  words = text.split()
  if len(words) < 5:
    return text
  new_words = []
  for w in words:
    if len(w) > 3 and w in w2v_wv and random.random() < replace_prob:
      similar_candidates = w2v_wv.most_similar(w, topn=2)
      if similar_candidates:
        new_words.append(similar_candidates[0][0])
      else:
        new_words.append(w)
    else:
      new_words.append(w)
  return ' '.join(new_words)


# 对 50% 的训练样本进行高质量语义扩充
aug_sample_size = int(len(X_train_raw) * 0.5)
aug_indices = random.sample(range(len(X_train_raw)), aug_sample_size)

augmented_texts = [
    w2v_augment_text(X_train_raw[i], w2v_model.wv) for i in aug_indices
]
augmented_labels = [y_train[i] for i in aug_indices]

# 组合训练集 (原始 + 增强)
X_train_combined = X_train_raw + augmented_texts
y_train_combined = np.concatenate([y_train, augmented_labels])

print(
    f'训练集扩充完毕: 原始 {len(X_train_raw)} 条 -> 扩充后'
    f' {len(X_train_combined)} 条 (新增 {len(augmented_texts)} 条增强样本)'
)

# ================= 4. 特征抽取与 MLP 对比训练 =================
print('\n>>> [4/5] 正在重新拟合 TF-IDF 并进行对照评测...')

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    max_features=12000,
    sublinear_tf=True,
)

# 1. 原始未增强特征
X_train_orig_tfidf = vectorizer.fit_transform(X_train_raw)
X_val_tfidf = vectorizer.transform(X_val_raw)

# 2. 扩充后的特征
vectorizer_aug = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    max_features=12000,
    sublinear_tf=True,
)
X_train_comb_tfidf = vectorizer_aug.fit_transform(X_train_combined)
X_val_aug_tfidf = vectorizer_aug.transform(X_val_raw)

# 最优冠军参数配置
champion_params = {
    'hidden_layer_sizes': (100,),
    'alpha': 0.0010,
    'early_stopping': True,
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 200,
    'random_state': SEED,
}

# 训练基准未增强模型
print('  • 正在训练 [未增强基线 MLP]...')
mlp_orig = MLPClassifier(**champion_params)
mlp_orig.fit(X_train_orig_tfidf, y_train)

orig_tr_preds = mlp_orig.predict(X_train_orig_tfidf)
orig_val_preds = mlp_orig.predict(X_val_tfidf)

orig_tr_acc = accuracy_score(y_train, orig_tr_preds)
orig_val_acc = accuracy_score(y_val, orig_val_preds)
orig_val_f1 = f1_score(y_val, orig_val_preds, average='macro')
orig_gap = orig_tr_acc - orig_val_acc

# 训练 Word2Vec 增强后的模型
print('  • 正在训练 [Word2Vec 增强后 MLP]...')
mlp_aug = MLPClassifier(**champion_params)
mlp_aug.fit(X_train_comb_tfidf, y_train_combined)

aug_tr_preds = mlp_aug.predict(X_train_comb_tfidf)
aug_val_preds = mlp_aug.predict(X_val_aug_tfidf)

aug_tr_acc = accuracy_score(y_train_combined, aug_tr_preds)
aug_val_acc = accuracy_score(y_val, aug_val_preds)
aug_val_f1 = f1_score(y_val, aug_val_preds, average='macro')
aug_gap = aug_tr_acc - aug_val_acc

# ================= 5. 汇总客观对比表格 (含训练集准确率与泛化差距) =================
print('\n' + '=' * 105)
print(
    '=== Word2Vec 数据增强对 MLP 分类性能与过拟合影响客观对比表 (含 Train Acc &'
    ' Gap) ==='
)
summary_df = pd.DataFrame([
    {
        '实验组别': '未增强基准组 (Original)',
        '训练样本数': len(X_train_raw),
        '收敛轮次': len(mlp_orig.loss_curve_),
        '最终Loss': mlp_orig.loss_curve_[-1],
        '训练集 Acc': orig_tr_acc,
        '验证集 Acc': orig_val_acc,
        '验证集 Macro-F1': orig_val_f1,
        '泛化差距(Gap)': orig_gap,
    },
    {
        '实验组别': 'Word2Vec 增强组 (Augmented)',
        '训练样本数': len(X_train_combined),
        '收敛轮次': len(mlp_aug.loss_curve_),
        '最终Loss': mlp_aug.loss_curve_[-1],
        '训练集 Acc': aug_tr_acc,
        '验证集 Acc': aug_val_acc,
        '验证集 Macro-F1': aug_val_f1,
        '泛化差距(Gap)': aug_gap,
    },
])
print(
    summary_df.to_string(
        index=False,
        justify='center',
        float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x,
    )
)
print('=' * 105)

# 动态保存对比柱状图 (同时对比训练集Acc与验证集F1)
fig, ax = plt.subplots(figsize=(8, 5))
x_idx = np.arange(len(summary_df))
width = 0.35

rects1 = ax.bar(
    x_idx - width / 2,
    summary_df['训练集 Acc'],
    width,
    label='训练集准确率 (Train Acc)',
    color='#d62728',
    alpha=0.85,
)
rects2 = ax.bar(
    x_idx + width / 2,
    summary_df['验证集 Macro-F1'],
    width,
    label='验证集宏平均F1 (Val Macro-F1)',
    color='#2ca02c',
    alpha=0.85,
)

ax.set_ylim(0.90, 1.02)
ax.set_title(
    'Word2Vec 数据增强对 MLP 拟合与泛化性能的影响对比',
    fontsize=12,
    pad=12,
    weight='bold',
)
ax.set_ylabel('评估得分 (Score)', fontsize=11)
ax.set_xticks(x_idx)
ax.set_xticklabels(summary_df['实验组别'], fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for rect in rects1:
  h = rect.get_height()
  ax.annotate(
      f'{h:.4f}',
      xy=(rect.get_x() + rect.get_width() / 2, h),
      xytext=(0, 2),
      textcoords='offset points',
      ha='center',
      va='bottom',
      fontsize=9,
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
      fontsize=9.5,
      weight='bold',
  )

ax.legend(loc='lower right', fontsize=10, frameon=True)
plt.tight_layout()

aug_fig_path = os.path.join(FIGURES_DIR, 'w2v_augmentation_comparison.png')
plt.savefig(aug_fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'\n✔ 包含拟合与泛化对比的高清柱状图已更新至: {aug_fig_path}')
print('>>> 数据增强对比全部运行完毕！')