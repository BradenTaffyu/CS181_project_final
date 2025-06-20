from __future__ import division
from math import log, exp
from operator import mul
from collections import Counter
import os
import matplotlib.pylab as pylab
import pickle as cPickle


class MyDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return 0

pos = MyDict()
neg = MyDict()
features = set()
totals = [0, 0]
delchars = ''.join(c for c in map(chr, range(128)) if not c.isalnum())

CDATA_FILE = "countdata.pickle"
FDATA_FILE = "reduceddata.pickle"


def negate_sequence(text):
    """
    Detects negations and transforms negated words into "not_" form.
    """
    negation = False
    delims = "?.,!:;"
    result = []
    words = text.split()
    prev = None
    pprev = None
    for word in words:
        # stripped = word.strip(delchars)
        stripped = word.strip(delims).lower()
        negated = "not_" + stripped if negation else stripped
        result.append(negated)
        if prev:
            bigram = prev + " " + negated
            result.append(bigram)
            if pprev:
                trigram = pprev + " " + bigram
                result.append(trigram)
            pprev = prev
        prev = negated

        if any(neg in word for neg in ["not", "n't", "no"]):
            negation = not negation

        if any(c in word for c in delims):
            negation = False

    return result


def train():
    global pos, neg, totals
    retrain = True
    
    # Load counts if they already exist.
    if not retrain and os.path.isfile(CDATA_FILE):
        #pos, neg, totals = cPickle.load(open(CDATA_FILE))
        with open(CDATA_FILE, 'rb') as f:
            pos, neg, totals = cPickle.load(f)
        return

    limit = 10000
    for file in os.listdir("./NB_on_Steam_Data/pos/")[:limit]:
        with open("./NB_on_Steam_Data/pos/" + file, 'r', encoding='utf-8') as f:
            for word in set(negate_sequence(f.read())):
                pos[word] += 1
                neg['not_' + word] += 1
    for file in os.listdir("./NB_on_Steam_Data/neg/")[:limit]:
        with open("./NB_on_Steam_Data/neg/" + file, 'r', encoding='utf-8') as f:
            for word in set(negate_sequence(f.read())):
                neg[word] += 1
                pos['not_' + word] += 1
    
    prune_features()

    totals[0] = sum(pos.values())
    totals[1] = sum(neg.values())
    
    countdata = (pos, neg, totals)
    #cPickle.dump(countdata, open(CDATA_FILE, 'w'))
    with open(CDATA_FILE, 'wb') as f:
        cPickle.dump(countdata, f)

def classify(text):
    words = set(word for word in negate_sequence(text) if word in features)
    if (len(words) == 0): return True
    # Probability that word occurs in pos documents
    pos_prob = sum(log((pos[word] + 1) / (2 * totals[0])) for word in words)
    neg_prob = sum(log((neg[word] + 1) / (2 * totals[1])) for word in words)
    return pos_prob > neg_prob

def classify2(text):
    """
    For classification from pretrained data
    """
    words = set(word for word in negate_sequence(text) if word in pos or word in neg)
    if (len(words) == 0): return True
    # Probability that word occurs in pos documents
    pos_prob = sum(log((pos[word] + 1) / (2 * totals[0])) for word in words)
    neg_prob = sum(log((neg[word] + 1) / (2 * totals[1])) for word in words)
    return pos_prob > neg_prob

def classify_demo(text):
    words = set(word for word in negate_sequence(text) if word in pos or word in neg)
    if (len(words) == 0): 
        print ("No features to compare on")
        return True

    pprob, nprob = 0, 0
    for word in words:
        pp = log((pos[word] + 1) / (2 * totals[0]))
        np = log((neg[word] + 1) / (2 * totals[1]))
        print ("%15s %.9f %.9f" % (word, exp(pp), exp(np)))
        pprob += pp
        nprob += np

    print ("Positive" if pprob > nprob else "Negative"), "log-diff = %.9f" % abs(pprob - nprob)

def MI(word):
    """
    Compute the weighted mutual information of a term.
    """
    T = totals[0] + totals[1]
    W = pos[word] + neg[word]
    I = 0
    if W==0:
        return 0
    if neg[word] > 0:
        # doesn't occur in -ve
        I += (totals[1] - neg[word]) / T * log ((totals[1] - neg[word]) * T / (T - W) / totals[1])
        # occurs in -ve
        I += neg[word] / T * log (neg[word] * T / W / totals[1])
    if pos[word] > 0:
        # doesn't occur in +ve
        I += (totals[0] - pos[word]) / T * log ((totals[0] - pos[word]) * T / (T - W) / totals[0])
        # occurs in +ve
        I += pos[word] / T * log (pos[word] * T / W / totals[0])
    return I

def get_relevant_features():
    pos_dump = MyDict({k: pos[k] for k in pos if k in features})
    neg_dump = MyDict({k: neg[k] for k in neg if k in features})
    totals_dump = [sum(pos_dump.values()), sum(neg_dump.values())]
    return (pos_dump, neg_dump, totals_dump)

def prune_features():
    """
    Remove features that appear only once.
    """
    global pos, neg
    for k in list(pos.keys()):              # <- 用 list() 拷贝一份键列表
        if pos[k] <= 1 and neg[k] <= 1:
            del pos[k]

    for k in list(neg.keys()):              # <- 同上
        if neg[k] <= 1 and pos[k] <= 1:
            del neg[k]


def feature_selection_trials():
    """
    Select top k features. Vary k and plot data
    """
    global pos, neg, totals, features
    retrain = True

    if not retrain and os.path.isfile(FDATA_FILE):
        pos, neg, totals = cPickle.load(open(FDATA_FILE))
        return

    words = list(set(pos.keys()) | set(neg.keys()))

    print ("Total no of features:", len(words))
    words.sort(key=lambda w: -MI(w))
    num_features, accuracy = [], []
    bestk = 0
    limit = 500
    path = "./NB_on_Steam_Data/"
    step = 200
    start = 20000
    best_accuracy = 0.0
    print(len(words))
    for w in words[:start]:
        features.add(w)
    for k in range(start, 60000, step):
        for w in words[k:k+step]:
            features.add(w)
        correct = 0
        size = 0

        for file in os.listdir(path + "pos")[:limit]:
            with open(path + "pos/" + file, 'r', encoding='utf-8') as f:
                correct += classify(f.read()) == True
            size += 1

        for file in os.listdir(path + "neg")[:limit]:
            with open(path + "neg/" + file, 'r', encoding='utf-8') as f:
                correct += classify(f.read()) == False
            size += 1

        num_features.append(k+step)
        accuracy.append(correct / size)
        if (correct / size) > best_accuracy:
            bestk = k
        print (k+step, correct / size)

    features = set(words[:bestk])
    #cPickle.dump(get_relevant_features(), open(FDATA_FILE, 'w'))
    with open(FDATA_FILE, 'wb') as f:
        cPickle.dump(get_relevant_features(), f)
    pylab.plot(num_features, accuracy)
    pylab.show()

def test_pang_lee():
    """
    Tests the Pang Lee dataset
    """
    total, correct = 0, 0
    for fname in os.listdir("txt_sentoken/pos"):
        with open("txt_sentoken/pos/" + fname, 'r', encoding='utf-8') as f:
            correct += int(classify2(f.read()) == True)
        total += 1
    for fname in os.listdir("txt_sentoken/neg"):
        with open("txt_sentoken/neg/" + fname, 'r', encoding='utf-8') as f:
            correct += int(classify2(f.read()) == False)
        total += 1
    print ("accuracy: %f" % (correct / total))

if __name__ == '__main__':
    train()
    feature_selection_trials()
    # test_pang_lee()
    # classify_demo(open("pos_example").read())
    # classify_demo(open("neg_example").read())
