import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
np.random.seed(SEED)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = CURRENT_DIR if not CURRENT_DIR.endswith("figures") else os.path.dirname(CURRENT_DIR)

# 1. 加载并清洗数据
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_data.csv'))

def clean_newsgroup_text(text):
    if not isinstance(text, str):
        return ""
    header_pattern = r'^(From|Lines|Reply-To|NNTP-Posting-Host|Organization|Distribution|Keywords|Path|Message-ID|Date|Article-I.D.|Sender|Followup-To):.*$'
    text = re.sub(header_pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\S+@\S+|http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text.lower()).strip()

clean_train = [clean_newsgroup_text(t) for t in train_df['text']]
y_all       = train_df['target'].values

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    clean_train, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

# 2. 特征维度消融实验 (控制变量法)
dims_to_test = [3000, 5000, 8000, 12000, 20000]
results = []

print(">>> 正在启动 TF-IDF 特征维度消融实验 (Ablation Study)...")

for d in dims_to_test:
    start_t = time.time()
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=d,
        sublinear_tf=True
    )
    X_tr = vectorizer.fit_transform(X_train_raw)
    X_va = vectorizer.transform(X_val_raw)
    
    # 采用标准分类器进行基准评测
    clf = LogisticRegression(C=100.0, max_iter=2000, random_state=SEED)
    clf.fit(X_tr, y_train)
    cost = time.time() - start_t
    
    tr_acc = accuracy_score(y_train, clf.predict(X_tr))
    v_acc  = accuracy_score(y_val, clf.predict(X_va))
    v_f1   = f1_score(y_val, clf.predict(X_va), average='macro')
    
    # 计算稀疏度
    sparsity = (1.0 - X_tr.nnz / (X_tr.shape[0] * X_tr.shape[1])) * 100
    
    results.append({
        "特征维度 (max_features)": d,
        "矩阵稀疏度": f"{sparsity:.2f}%",
        "训练集 Acc": tr_acc,
        "验证集 Acc": v_acc,
        "验证集 Macro-F1": v_f1,
        "泛化差距 (Gap)": tr_acc - v_acc,
        "耗时(秒)": cost
    })
    print(f"  • max_features={d:<5} 测试完成 -> Val F1: {v_f1:.4f}, 耗时: {cost:.2f}s")

res_df = pd.DataFrame(results)

print("\n" + "="*85)
print("=== TF-IDF 特征维度 (max_features) 消融实验实测对比表 ===")
print(res_df.to_string(index=False, justify='center', float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
print("="*85)