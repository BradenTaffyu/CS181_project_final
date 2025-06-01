import numpy as np
from collections import defaultdict
import math

def calculate_tfidf(cluster_texts, all_texts):
    """计算TF-IDF值"""
    # 统计词频
    cluster_tf = defaultdict(int)
    global_df = defaultdict(int)
    
    for text in cluster_texts:
        words = set(text.split())
        for word in words:
            cluster_tf[word] += 1
    
    for text in all_texts:
        words = set(text.split())
        for word in words:
            global_df[word] += 1
    
    # 计算TF-IDF
    tfidf = {}
    cluster_size = len(cluster_texts)
    total_docs = len(all_texts)
    
    for word, tf in cluster_tf.items():
        idf = math.log((total_docs + 1) / (global_df.get(word, 0) + 1))
        tfidf[word] = (tf / cluster_size) * idf
    
    return tfidf


def calculate_kl_divergence(cluster_dist, global_dist):
    """计算KL散度差异"""
    kl_scores = {}
    epsilon = 1e-9  # 防止除以0
    
    for word, p in cluster_dist.items():
        q = global_dist.get(word, epsilon)
        kl = p * math.log(p / q)
        kl_scores[word] = kl
    
    return kl_scores