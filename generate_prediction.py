import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# ================= 0. 严格锁定随机数种子 =================
SEED = 42
np.random.seed(SEED)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = CURRENT_DIR if not CURRENT_DIR.endswith("figures") else os.path.dirname(CURRENT_DIR)

# ================= 1. 加载全量训练集与测试集 =================
print(">>> [1/3] 正在加载训练数据与待预测测试数据...")
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_data.csv'))
test_df  = pd.read_csv(os.path.join(BASE_DIR, 'test_data_unlabeled.csv'))

def clean_newsgroup_text(text):
    if not isinstance(text, str):
        return ""
    header_pattern = r'^(From|Lines|Reply-To|NNTP-Posting-Host|Organization|Distribution|Keywords|Path|Message-ID|Date|Article-I.D.|Sender|Followup-To):.*$'
    text = re.sub(header_pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\S+@\S+|http\S+|www\S+', '', text)
    # 保留字母与数字
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text.lower()).strip()

clean_train = [clean_newsgroup_text(t) for t in train_df['text']]
clean_test  = [clean_newsgroup_text(t) for t in test_df['text']]
y_train     = train_df['target'].values

print(f"数据加载完毕: 训练样本 {len(clean_train)} 条, 待预测测试样本 {len(clean_test)} 条")

# ================= 2. 全量 TF-IDF 特征工程 (12000 维) =================
print(">>> [2/3] 正在构建全量 TF-IDF 特征工程矩阵...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    max_features=12000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(clean_train)
X_test_tfidf  = vectorizer.transform(clean_test)

# ================= 3. 使用最终选定的最优模型训练并生成预测 =================
print(">>> [3/3] 使用最终冠军配置 (MLP: Hidden=100, 早停, alpha=0.0010) 进行全量拟合与推理...")
champion_model = MLPClassifier(
    hidden_layer_sizes=(100,),
    alpha=0.0010,
    early_stopping=True,
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=SEED
)

champion_model.fit(X_train_tfidf, y_train)
test_predictions = champion_model.predict(X_test_tfidf)

path_without_s = os.path.join(BASE_DIR, 'prediction.csv')
path_with_s    = os.path.join(BASE_DIR, 'predictions.csv')

pd.DataFrame(test_predictions).to_csv(path_without_s, index=False, header=False)
pd.DataFrame(test_predictions).to_csv(path_with_s, index=False, header=False)

print("\n" + "="*70)
print("✔ 预测成功完成！")
print(f"✔ 导出文件行数: {len(test_predictions)} 行 (与测试集 2457 条完全对齐)")
print(f"✔ 文件已生成为: {path_without_s} (符合最新群公告命名)")
print(f"✔ 备份文件已生成为: {path_with_s} (符合课件原本命名)")
print("="*70)