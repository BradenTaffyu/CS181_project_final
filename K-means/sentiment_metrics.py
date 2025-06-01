import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np
nltk.download('stopwords', quiet=True)

def calculate_relative_frequency(cluster_texts, global_texts, existing_labels=[]):
    """实现相对词频算法 score = count.x * log(count.x/count.y)"""
    # 停用词过滤
    stop_words = set(stopwords.words('english'))
    
    # 统计簇内词频
    cluster_counter = defaultdict(int)
    for text in cluster_texts:
        words = [word for word in text.split() if word not in stop_words]
        for word in words:
            cluster_counter[word] += 1
    
    # 统计全局词频
    global_counter = defaultdict(int)
    for text in global_texts:
        words = [word for word in text.split() if word not in stop_words]
        for word in words:
            global_counter[word] += 1
    
    # 创建DataFrame并合并
    cluster_df = pd.DataFrame(list(cluster_counter.items()), columns=['word', 'count_x'])
    global_df = pd.DataFrame(list(global_counter.items()), columns=['word', 'count_y'])
    merged = pd.merge(cluster_df, global_df, on='word', how='inner')
    
    # 计算相对词频得分
    merged['score'] = merged['count_x'] * np.log(merged['count_x'] / merged['count_y'].astype(float))
    
    filtered_scores = [(row.word, float(row.score)) for row in merged.itertuples(index=False) 
                      if row.word not in existing_labels and row.word not in stop_words]
    return pd.DataFrame(filtered_scores, columns=['word', 'score']).sort_values('score', ascending=False)