"""
F-Score metrics for testing classifier, also includes functions for data extraction.
Author: Vivek Narayanan
"""
import os
from info import MyDict
import info, pickle

def get_paths():
    """
    Returns supervised paths annotated with their actual labels.
    """
    posfiles = [("./aclImdb/test/pos/" + f, True) for f in os.listdir("./aclImdb/test/pos/")]
    negfiles = [("./aclImdb/test/neg/" + f, False) for f in os.listdir("./aclImdb/test/neg/")]
    return posfiles + negfiles


def fscore(classifier, file_paths):
    tpos, fpos, fneg, tneg = 0, 0, 0, 0
    for path, label in file_paths:
        result = classifier(open(path,encoding='utf-8').read())
        if label and result:
            tpos += 1
        elif label and (not result):
            fneg += 1
        elif (not label) and result:
            fpos += 1
        else:
            tneg += 1
    prec = 1.0 * tpos / (tpos + fpos)
    recall = 1.0 * tpos / (tpos + fneg)
    f1 = 2 * prec * recall / (prec + recall)
    accu = 100.0 * (tpos + tneg) / (tpos+tneg+fpos+fneg)
    print ("True Positives: %d\nFalse Positives: %d\nFalse Negatives: %d\n" % (tpos, fpos, fneg))
    print ("Precision: %lf\nRecall: %lf\nAccuracy: %lf" % (prec, recall, accu))

def main():
    from info import classify, train 
    with open("reduceddata.pickle", "rb") as f:
        info.pos, info.neg, info.totals = pickle.load(f)
    # 还要把 features 从 info.feature_selection_trials 里拿出或者重跑一次
    info.features = set(info.pos.keys())  # 或者重新执行 feature_selection_trials，取 info.features
    # 然后再跑 fscore
    paths = get_paths()
    fscore(info.classify, paths)

if __name__ == '__main__':
    main()
