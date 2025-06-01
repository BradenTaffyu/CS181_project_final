import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 读取CSV文件（添加错误处理参数）
df = pd.read_csv('Sentiment Analysis Dataset.csv', 
                on_bad_lines='skip',  # 跳过格式错误行
                engine='python',       # 确保更好的错误处理
                quoting=3)             # 处理带引号的内容

neg_df = df[df['Sentiment'] == 0]
pos_df = df[df['Sentiment'] == 1]

# 划分训练集、测试集和无标签集
# 首先从正负样本中各抽取一部分作为无标签数据
neg_unlabeled, neg_labeled = train_test_split(neg_df, test_size=0.8, random_state=42)
pos_unlabeled, pos_labeled = train_test_split(pos_df, test_size=0.8, random_state=42)

# 再将有标签数据划分为训练集和测试集
neg_train, neg_test = train_test_split(neg_labeled, test_size=0.25, random_state=42)  # 0.8 * 0.25 = 0.2 (总数据的20%)
pos_train, pos_test = train_test_split(pos_labeled, test_size=0.25, random_state=42)  # 0.8 * 0.25 = 0.2 (总数据的20%)

# 创建目录结构
base_dir = 'aclImdb'
dirs = [
    (base_dir, 'train', 'neg'),
    (base_dir, 'train', 'pos'),
    (base_dir, 'test', 'neg'),
    (base_dir, 'test', 'pos'),
    (base_dir, 'unlabeled'),  # 新增无标签文件夹
]
for parts in dirs:
    os.makedirs(os.path.join(*parts), exist_ok=True)

# 保存训练集neg
train_neg_dir = os.path.join(base_dir, 'train', 'neg')
for idx, row in neg_train.iterrows():
    with open(os.path.join(train_neg_dir, f"{row['ItemID']}.txt"), 'w', encoding='utf-8') as f:
        f.write(row['SentimentText'])

# 保存训练集pos
train_pos_dir = os.path.join(base_dir, 'train', 'pos')
for idx, row in pos_train.iterrows():
    with open(os.path.join(train_pos_dir, f"{row['ItemID']}.txt"), 'w', encoding='utf-8') as f:
        f.write(row['SentimentText'])

# 保存测试集neg
test_neg_dir = os.path.join(base_dir, 'test', 'neg')
for idx, row in neg_test.iterrows():
    with open(os.path.join(test_neg_dir, f"{row['ItemID']}.txt"), 'w', encoding='utf-8') as f:
        f.write(row['SentimentText'])

# 保存测试集pos
test_pos_dir = os.path.join(base_dir, 'test', 'pos')
for idx, row in pos_test.iterrows():
    with open(os.path.join(test_pos_dir, f"{row['ItemID']}.txt"), 'w', encoding='utf-8') as f:
        f.write(row['SentimentText'])

# 保存无标签数据（忽略Sentiment列）
unlabeled_dir = os.path.join(base_dir, 'unlabeled')
for idx, row in pd.concat([neg_unlabeled, pos_unlabeled]).iterrows():
    with open(os.path.join(unlabeled_dir, f"{row['ItemID']}.txt"), 'w', encoding='utf-8') as f:
        f.write(row['SentimentText'])

print("数据划分和保存完成。")