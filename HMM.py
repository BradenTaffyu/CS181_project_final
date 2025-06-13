import os
import sys
import numpy as np
from hmmlearn import hmm
import pickle

class MyDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return 0

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 确保项目根目录在sys.path中
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import metric_modified as metric
import NB

# 数据集路径配置
DATA_PATHS = {
    'train_pos': 'pos',  
    'train_neg': 'neg',  
    'test_pos': 'test_pos',  
    'test_neg': 'test_neg'   
}

from NB import negate_sequence


def load_texts_labels(base_path, data_type='train'):

    texts, labels = [], []
    prefix = 'train' if data_type == 'train' else 'test'
    for pos_neg in ['pos', 'neg']:
        data_type_key = f'{prefix}_{pos_neg}'
        label = pos_neg == 'pos'
        folder = os.path.join(base_path, DATA_PATHS[data_type_key])
        if not os.path.exists(folder):
            print(f"Warning: Directory not found: {folder}")
            continue
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels


def build_vocab(top_k=None):
    """
    Build vocabulary from NB-trained word counts
    Args:
        top_k: 最多选择的特征词数量
    """
    # 收集所有词
    words = list(set(NB.pos.keys()) | set(NB.neg.keys()))
    
    # 如果设置了top_k，限制特征数量
    if top_k is not None:
        words = words[:top_k]
    
    # 添加未知词标记
    words.append('<UNK>')
    return {w: i for i, w in enumerate(words)}


def texts_to_sequences(texts, w2i):
    """
    Convert list of texts to concatenated observation sequence and lengths
    """
    seqs, lengths = [], []
    for txt in texts:
        obs = [w2i.get(w, w2i['<UNK>']) for w in negate_sequence(txt)]
        seqs.extend(obs)
        lengths.append(len(obs))
    X = np.array(seqs).reshape(-1, 1)
    return X, lengths


def train_hmm(X, lengths, n_states=5, n_iter=25, n_features=None):
    """
    Train a discrete-observation HMM (prefer CategoricalHMM) on the given data
    """
    # Prefer CategoricalHMM for integer-coded symbols if available
    # 打印出当前迭代次数，对数似然值，每次迭代的改进值
    if hasattr(hmm, 'CategoricalHMM'):
        try:
            model = hmm.CategoricalHMM(n_components=n_states, n_iter=n_iter, verbose=True)
        except TypeError:
            # Older API may expect n_features argument
            model = hmm.CategoricalHMM(n_components=n_states, n_features=n_features, n_iter=n_iter, verbose=True)
    else:
        # Fallback to MultinomialHMM (legacy behavior)
        model = hmm.MultinomialHMM(n_components=n_states, n_iter=n_iter, verbose=True)
        if n_features is not None and hasattr(model, 'n_features'):
            model.n_features = n_features
    # Explicitly set number of features if supported
    if n_features is not None and hasattr(model, 'n_features'):
        model.n_features = n_features
    # Fit the model on the observation sequences
    model.fit(X, lengths)
    return model


def map_states_to_labels(model, X, lengths, labels):
    """
    Determine which hidden state corresponds to positive sentiment
    """
    states = model.predict(X, lengths)
    # Count occurrences of each state for positive vs negative
    state_pos_count = {s: 0 for s in range(model.n_components)}
    state_total_count = {s: 0 for s in range(model.n_components)}
    idx = 0
    for seq_i, length in enumerate(lengths):
        label = labels[seq_i]
        seq_states = states[idx:idx + length]
        idx += length
        for s in seq_states:
            state_total_count[s] += 1
            if label:
                state_pos_count[s] += 1
    # Compute positive ratio per state
    state_scores = {s: (state_pos_count[s] / state_total_count[s]) if state_total_count[s] > 0 else 0
                    for s in state_total_count}
    # The state with highest positive ratio is the positive state
    return max(state_scores, key=state_scores.get)


def make_hmm_classifier(model, w2i, state_pos):
    """
    Return a function that classifies a text as positive if majority Viterbi state equals state_pos
    """
    def classify(text):
        # Preprocess and handle empty sequences
        tokens = negate_sequence(text)
        if not tokens:
            # Default to positive if no observable tokens
            return True
        obs = np.array([w2i.get(w, w2i['<UNK>']) for w in tokens]).reshape(-1, 1)
        states = model.predict(obs)
        counts = np.bincount(states, minlength=model.n_components)
        pred_state = np.argmax(counts)
        return pred_state == state_pos
    return classify


if __name__ == '__main__':
    # 1) Train NB to populate info.pos and info.neg
    NB.train()
    
    # 2) Load training data
    base_path = script_dir
    train_texts, train_labels = load_texts_labels(base_path, 'train')
    
    # 3) Build vocabulary from NB counts
    TOP_K = 30000  # 控制词汇表大小
    w2i = build_vocab(top_k=TOP_K)
    n_features = len(w2i)
    
    # 4) Convert texts to sequences, filtering out empty sequences
    seqs, lengths_train, labels_train = [], [], []
    for txt, lab in zip(train_texts, train_labels):
        obs = [w2i.get(w, w2i['<UNK>']) for w in negate_sequence(txt)]
        if not obs:
            continue
        seqs.extend(obs)
        lengths_train.append(len(obs))
        labels_train.append(lab)
    X_train = np.array(seqs).reshape(-1, 1)
    
    # 5) Train class-specific HMMs on positive and negative sequences
    pos_seqs, pos_lens = [], []
    neg_seqs, neg_lens = [], []
    idx = 0
    for length, lab in zip(lengths_train, labels_train):
        slice_obs = seqs[idx:idx + length]
        idx += length
        if lab:
            pos_seqs.extend(slice_obs)
            pos_lens.append(length)
        else:
            neg_seqs.extend(slice_obs)
            neg_lens.append(length)
    X_pos = np.array(pos_seqs).reshape(-1, 1)
    X_neg = np.array(neg_seqs).reshape(-1, 1)
    N_STATES = 4
    model_pos = train_hmm(X_pos, pos_lens, n_states=N_STATES, n_iter=25, n_features=n_features)
    model_neg = train_hmm(X_neg, neg_lens, n_states=N_STATES, n_iter=25, n_features=n_features)

    # 保存模型和词汇表
    model_data = {
        'model_pos': model_pos,
        'model_neg': model_neg,
        'w2i': w2i,
        'n_features': n_features
    }
    
    model_path = os.path.join(script_dir, 'hmm_model.pickle')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"模型已保存到: {model_path}")

    # 6) Define classifier by comparing log-likelihoods under each HMM
    def classify(text):
        tokens = negate_sequence(text)
        if not tokens:
            return True
        obs = np.array([w2i.get(w, w2i['<UNK>']) for w in tokens]).reshape(-1, 1)
        try:
            pos_score = model_pos.score(obs)
            neg_score = model_neg.score(obs)
        except Exception:
            return True
        return pos_score > neg_score

    # 7) Evaluate on test set
    test_texts, test_labels = load_texts_labels(base_path, 'test')
    
    # 计算测试集上的性能指标
    tpos, fpos, fneg, tneg = 0, 0, 0, 0
    for text, label in zip(test_texts, test_labels):
        result = classify(text)
        if label and result:
            tpos += 1
        elif label and (not result):
            fneg += 1
        elif (not label) and result:
            fpos += 1
        else:
            tneg += 1
    
    # 计算性能指标
    prec = 1.0 * tpos / (tpos + fpos) if (tpos + fpos) > 0 else 0
    recall = 1.0 * tpos / (tpos + fneg) if (tpos + fneg) > 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
    accu = 100.0 * (tpos + tneg) / (tpos + tneg + fpos + fneg)
    
    print("Test Results:")
    print("True Positives: %d\nFalse Positives: %d\nFalse Negatives: %d\n" % (tpos, fpos, fneg))
    print("Precision: %lf\nRecall: %lf\nF1 Score: %lf\nAccuracy: %lf" % (prec, recall, f1, accu))
    print("tpos,tneg,fpos,fneg:", tpos, tneg, fpos, fneg)