"""
F-Score metrics for testing classifier, also includes functions for data extraction.
Author: Vivek Narayanan
"""
import os
from NB import MyDict
import NB, pickle

def get_paths():
    """
    Returns supervised paths annotated with their actual labels.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    test_pos_dir = os.path.join(base_dir, "test_pos")
    test_neg_dir = os.path.join(base_dir, "test_neg")
    
    posfiles = [(os.path.join(test_pos_dir, f), True) for f in os.listdir(test_pos_dir)]
    negfiles = [(os.path.join(test_neg_dir, f), False) for f in os.listdir(test_neg_dir)]
    return posfiles + negfiles


def fscore(classifier, file_paths):
    tpos, fpos, fneg, tneg = 0, 0, 0, 0
    for path, label in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            result = classifier(f.read())
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
    print("tpos,tneg,fpos,fneg:", tpos, tneg, fpos, fneg)

def main():
    from NB import classify_with_features, train 
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(base_dir, "reduceddata.pickle")
    

    with open(pickle_path, "rb") as f:
        NB.pos, NB.neg, NB.totals = pickle.load(f)
    NB.features = set(NB.pos.keys())  
    paths = get_paths()
    fscore(NB.classify_with_features, paths)

if __name__ == '__main__':
    main()
