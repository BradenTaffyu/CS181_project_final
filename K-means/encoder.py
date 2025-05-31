import re
from pathlib import Path
import pandas as pd
from clean_data import clean_data
import tensorflow_hub as hub
import numpy as np

unsup_dir = Path('../aclImdb') / 'train' / 'unsup'
if not unsup_dir.exists():
    raise FileNotFoundError(f"Directory not found: {unsup_dir}")
unsup_texts = []
for file_path in unsup_dir.glob('*.txt'):
    raw = file_path.read_text(encoding='utf-8')
    unsup_texts.append(clean_data(raw))

print(f"Loaded {len(unsup_texts)} unlabeled samples from {unsup_dir}")

encoder_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
# use local model instead
encoder_path = "../universal-sentence-encoder-tensorflow2-large-v2/"
use_large = hub.load(encoder_path)

# Encode the data in batches to avoid memory issues


batch_size = 128
all_embeddings = []
for i in range(0, len(unsup_texts), batch_size):
    batch = unsup_texts[i:i+batch_size]
    batch_embeddings = use_large(batch)
    all_embeddings.append(batch_embeddings.numpy())
embeddings = np.vstack(all_embeddings)

print(embeddings.shape)
