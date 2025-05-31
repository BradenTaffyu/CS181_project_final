import re
from pathlib import Path
import pandas as pd
from clean_data import clean_data
import tensorflow_hub as hub
import numpy as np
from sentence_transformers import SentenceTransformer
import json


unsup_dir = Path('../aclImdb') / 'train' / 'unsup'
if not unsup_dir.exists():
    raise FileNotFoundError(f"Directory not found: {unsup_dir}")

# 存储文本和对应的文件名
unsup_texts = []
file_mapping = []  # 只保存文件名和索引

for i, file_path in enumerate(unsup_dir.glob('*.txt')):
    raw = file_path.read_text(encoding='utf-8')
    cleaned_text = clean_data(raw)
    unsup_texts.append(cleaned_text)
    
    # 只保存必要的文件信息
    file_mapping.append({
        'index': i,
        'filename': file_path.name,
        'file_path': str(file_path)
    })

print(f"Loaded {len(unsup_texts)} unlabeled samples from {unsup_dir}")

# encoder_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
# # use local model instead
# encoder_path = "../universal-sentence-encoder-tensorflow2-large-v2/"
model = SentenceTransformer('all-MiniLM-L6-v2')
# use_large = hub.load(encoder_path)

# Encode the data in batches to avoid memory issues
batch_size = 128
all_embeddings = []

for i in range(0, len(unsup_texts), batch_size):
    batch = unsup_texts[i:i+batch_size]
    batch_embeddings = model.encode(batch)
    all_embeddings.append(batch_embeddings)
    print(f"处理批次 {i//batch_size + 1}, 样本 {i+1}-{min(i+batch_size, len(unsup_texts))}")

embeddings = np.vstack(all_embeddings)

print(embeddings.shape)

# 保存embedding
np.save('unsup_embeddings.npy', embeddings)

# 只保存文件名映射（简化版）
with open('file_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(file_mapping, f, ensure_ascii=False, indent=2)

print(f"已保存:")
print(f"  - Embeddings: unsup_embeddings.npy (shape: {embeddings.shape})")
print(f"  - File mapping: file_mapping.json ({len(file_mapping)} files)")

print(f"\n统计信息:")
print(f"  - 总样本数: {len(unsup_texts)}")
print(f"  - Embedding维度: {embeddings.shape[1]}")
