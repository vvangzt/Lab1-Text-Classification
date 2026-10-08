import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import Word2Vec

# ================= 0. 全局配置锁定 =================
SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= 1. 加载并清洗数据 =================
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

# ================= 2. 训练 Word2Vec 词向量 =================
print(">>> 正在训练 Word2Vec (100维)...")
tokenized_train = [t.split() for t in X_train_raw]
w2v = Word2Vec(sentences=tokenized_train, vector_size=100, window=5, min_count=2, workers=4, seed=SEED)

# ================= 3. 计算文档均值向量 (Average Pooling: 文本 -> 100维稠密向量) =================
print(">>> 正在将每篇新闻转换为 100 维的纯 Word2Vec 稠密文档向量...")

def text_to_w2v_vector(text, model, dim=100):
    words = [w for w in text.split() if w in model.wv]
    if len(words) == 0:
        return np.zeros(dim)
    # 取这篇新闻所有词向量的平均值
    return np.mean(model.wv[words], axis=0)

X_train_w2v = np.array([text_to_w2v_vector(t, w2v) for t in X_train_raw])
X_val_w2v   = np.array([text_to_w2v_vector(t, w2v) for t in X_val_raw])

print(f"纯 Word2Vec 特征维度: 训练集 {X_train_w2v.shape}, 验证集 {X_val_w2v.shape} (低维稠密！)")

# ================= 4. 用纯 Word2Vec 训练 MLP =================
print(">>> 正在用纯 Word2Vec 稠密特征训练 MLP...")
mlp_w2v = MLPClassifier(
    hidden_layer_sizes=(100,),
    alpha=0.0010,
    early_stopping=True,
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=SEED
)

mlp_w2v.fit(X_train_w2v, y_train)

val_preds = mlp_w2v.predict(X_val_w2v)
w2v_acc = accuracy_score(y_val, val_preds)
w2v_f1  = f1_score(y_val, val_preds, average='macro')

# ================= 5. 三种方案终极对比表 =================
print("\n" + "="*85)
print("=== 文本表征方法终极对决：纯 TF-IDF vs. 纯 Word2Vec vs. TF-IDF+增强 ===")
comparison_table = pd.DataFrame([
    {
        "表征方案": "纯 Word2Vec 均值向量 (稠密表征)",
        "输入维度": "100 维 (稠密)",
        "验证集 Acc": w2v_acc,
        "验证集 Macro-F1": w2v_f1,
        "特性分析": "低维稠密、语义连续，但关键词易被平均稀释"
    },
    {
        "表征方案": "纯 TF-IDF 向量 (高维稀疏表征)",
        "输入维度": "12,000 维 (稀疏)",
        "验证集 Acc": 0.9376,
        "验证集 Macro-F1": 0.9383,
        "特性分析": "高维稀疏，关键词突出，主题区分度极高"
    },
    {
        "表征方案": "TF-IDF + Word2Vec 近义词数据增强",
        "输入维度": "12,000 维 (扩充样本)",
        "验证集 Acc": 0.9389,
        "验证集 Macro-F1": 0.9395,
        "特性分析": "融合语义多样性与高维关键词，全场综合最优"
    }
])
print(comparison_table.to_string(index=False, justify='center', float_format=lambda x: f"{x:.4f}"))
print("="*85)