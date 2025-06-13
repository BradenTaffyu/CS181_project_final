import os
import sys
import numpy as np
from hmmlearn import hmm
class MyDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return 0
# Ensure project root is in sys.path for importing metric module
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import metric
import info
from info import negate_sequence


def load_texts_labels(base_path):
    """
    Load all review texts and their labels (True for positive, False for negative)
    """
    texts, labels = [], []
    for subdir, label in [('pos', True), ('neg', False)]:
        folder = os.path.join(base_path, subdir)
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels


def build_vocab(top_k=None):
    """
    Build vocabulary from NB-trained word counts, optionally selecting top_k by mutual information.
    """
    # Collect all words from NB counts
    words = list(set(info.pos.keys()) | set(info.neg.keys()))
    # If requested, select top_k words by MI score
    if top_k is not None:
        words.sort(key=lambda w: info.MI(w), reverse=True)
        words = words[:top_k]
    # Add unknown token
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


def train_hmm(X, lengths, n_states=3, n_iter=25, n_features=None):
    """
    Train a discrete-observation HMM (prefer CategoricalHMM) on the given data
    """
    # Prefer CategoricalHMM for integer-coded symbols if available
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
    info.train()
    # 2) Load training data
    base_path = os.path.dirname(__file__)
    texts, labels = load_texts_labels(base_path)
    # 3) Build vocabulary from NB counts and MI-based selection
    TOP_K = 30000  # adjust this value to control vocabulary size
    w2i = build_vocab(top_k=TOP_K)
    n_features = len(w2i)
    # 4) Convert texts to sequences, filtering out empty sequences
    seqs, lengths_train, labels_train = [], [], []
    for txt, lab in zip(texts, labels):
        obs = [w2i.get(w, w2i['<UNK>']) for w in negate_sequence(txt)]
        if not obs:
            continue
        seqs.extend(obs)
        lengths_train.append(len(obs))
        labels_train.append(lab)
    X_train = np.array(seqs).reshape(-1, 1)
    # 5) Train class-specific HMMs on positive and negative sequences (use 3 states)
    # Split the concatenated sequences back into class-wise sequences
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
    N_STATES = 3
    model_pos = train_hmm(X_pos, pos_lens, n_states=N_STATES, n_iter=25, n_features=n_features)
    model_neg = train_hmm(X_neg, neg_lens, n_states=N_STATES, n_iter=25, n_features=n_features)

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
    test_paths = metric.get_paths()
    metric.fscore(classify, test_paths)
