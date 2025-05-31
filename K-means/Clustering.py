import numpy as np
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from clean_data import clean_data


def load_data():
    """加载embedding和文件映射"""
    embeddings = np.load('K-means\\unsup_embeddings.npy')
    
    with open('file_mapping.json', 'r', encoding='utf-8') as f:
        file_mapping = json.load(f)
    
    print(f"加载了 {embeddings.shape[0]} 个样本，embedding维度: {embeddings.shape[1]}")
    return embeddings, file_mapping


def read_file_content(file_path, max_length=200):
    """根据文件路径读取原始内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 返回清理后的内容（截取前max_length个字符）
        cleaned = clean_data(content)
        if len(cleaned) > max_length:
            return cleaned[:max_length] + "..."
        return cleaned
    except Exception as e:
        return f"读取文件失败: {e}"


def kmeans_clustering(embeddings, k):
    """执行K-means聚类"""
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans

import sys
from pathlib import Path

# 添加项目根目录到系统路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))  # 添加当前目录的上级目录到路径



def analyze_clusters(embeddings, file_mapping, cluster_labels, k):

    # 在 analyze_clusters 开头添加测试
    test_texts = [
    "This movie is absolutely wonderful!",  # 积极样本
    "Terrible acting and boring plot",      # 消极样本
    ]

    print("\n情感模型验证:")
    for text in test_texts:
        result = info.classify(clean_data(text))
        print(f"文本: {text[:30]}... → 预测结果: {'积极' if result else '消极'}")

    """分析聚类结果"""
    print(f"\n=== 聚类分析 (k={k}) ===")

    cluster_stats = defaultdict(lambda: {
        'sentiment_scores': [],
        'sentiment_counts': {'positive': 0, 'negative': 0}
    })

    for cluster_id in range(k):
        # 找到属于当前簇的样本索引
        indices = np.where(cluster_labels == cluster_id)[0]
        print(f"\n--- 簇 {cluster_id} ({len(indices)} 个样本) ---")
        
        for idx in indices:
            file_path = file_mapping[idx]['file_path']
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                cleaned_text = clean_data(text)
                # 情感分析（需要根据实际模型调整阈值）
                is_positive = info.classify(cleaned_text)
                sentiment_score = 1 if is_positive else -1  # 二元分类得分
                
                # 更新统计
                cluster_stats[cluster_id]['sentiment_scores'].append(sentiment_score)
                cluster_stats[cluster_id]['sentiment_counts']['positive' if is_positive else 'negative'] += 1
            except Exception as e:
                print(f"文件读取失败: {file_path} - {str(e)}")
                continue
    for cluster_id in range(k):
        stats = cluster_stats[cluster_id]
        total = len(stats['sentiment_scores'])

        if total == 0:  # 添加空值保护
            print(f"\n--- 簇 {cluster_id} 无有效样本 ---")
            continue
        
        # 计算平均情感得分
        avg_score = np.mean(stats['sentiment_scores']) 
        
        # 计算情感分布比例
        pos_ratio = stats['sentiment_counts']['positive'] / total * 100
        neg_ratio = stats['sentiment_counts']['negative'] / total * 100
        
        print(f"\n--- 簇 {cluster_id} 情感分析 ---")
        print(f"平均情感得分: {avg_score:.2f} (范围: [-1.0, 1.0])")
        print(f"情感分布:")
        print(f"  积极: {pos_ratio:.1f}%")
        print(f"  消极: {neg_ratio:.1f}%")

        # 显示前几个样本的文件名和内容预览
        print("样本文件:")
        for i, idx in enumerate(indices[:10]):
            filename = file_mapping[idx]['filename']
            file_path = file_mapping[idx]['file_path']
            content_preview = read_file_content(file_path, max_length=100)
            print(f"  {filename}: {content_preview}")
        
        if len(indices) > 5:
            print(f"  ... 还有 {len(indices)-10} 个样本")


def save_clustering_results(file_mapping, cluster_labels, k):
    """保存聚类结果"""
    # 将聚类结果添加到文件映射中
    results = []
    for i, mapping in enumerate(file_mapping):
        result = mapping.copy()
        result['cluster'] = int(cluster_labels[i])
        results.append(result)
    
    # 保存结果
    with open(f'clustering_results_k{k}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"聚类结果已保存: clustering_results_k{k}.json")
    return results


def visualize_clusters(embeddings, cluster_labels, k):
    """可视化聚类结果"""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap='tab10', s=10, alpha=0.7)
    plt.title(f'K-means聚类结果 (k={k})')
    plt.xlabel(f'PCA 1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PCA 2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
    plt.colorbar(scatter, ticks=range(k))
    plt.tight_layout()
    plt.savefig(f'clustering_k{k}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 加载数据
    embeddings, file_mapping = load_data()
    
    # 设置聚类数量
    k = 3  # 你可以修改这个值
    from collections import defaultdict
    import pickle
    import info
    from info import classify, MyDict

    with open("reduceddata.pickle", "rb") as f:
        info.pos, info.neg, info.totals = pickle.load(f)

    info.features = set(info.pos.keys())|set(info.neg.keys())  

    # 执行K-means聚类
    cluster_labels, kmeans = kmeans_clustering(embeddings, k)
    print(f"聚类完成，k={k}")
    
    # 分析聚类结果
    analyze_clusters(embeddings, file_mapping, cluster_labels, k)
    
    # 保存结果
    results = save_clustering_results(file_mapping, cluster_labels, k)
    
    # 可视化
    visualize_clusters(embeddings, cluster_labels, k)
    
    # 显示每个簇的文件数量统计
    print(f"\n=== 簇大小统计 ===")
    for cluster_id in range(k):
        count = np.sum(cluster_labels == cluster_id)
        percentage = count / len(cluster_labels) * 100
        print(f"簇 {cluster_id}: {count} 个文件 ({percentage:.1f}%)")