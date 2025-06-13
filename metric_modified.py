"""
F-Score metrics for testing classifier, also includes functions for data extraction.
Author: Vivek Narayanan
"""
import os
from NB import MyDict
import NB, pickle
import csv


def get_paths(game_name):
    """
    Returns supervised paths annotated with their actual labels.
    """
    # 使用相对于脚本文件的路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "Data")
    csv_file = os.path.join(data_dir, f"{game_name}.csv")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        
    return [(csv_file, True)] if os.path.exists(csv_file) else []

def fscore(classifier, file_paths):
    """
    计算分类器的F-score指标
    Args:
        classifier: 分类器函数
        file_paths: 包含(文件路径,标签)元组的列表
    """
    tpos, fpos, fneg, tneg = 0, 0, 0, 0
    
    for path, label in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue
            
        with open(path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                review_text = row.get('review', '')
                result = classifier(review_text)
                
                if label and result:
                    tpos += 1
                elif label and (not result):
                    fneg += 1
                elif (not label) and result:
                    fpos += 1
                else:
                    tneg += 1
    
    if tpos + fpos == 0:
        prec = 0
    else:
        prec = 1.0 * tpos / (tpos + fpos)
        
    if tpos + fneg == 0:
        recall = 0
    else:
        recall = 1.0 * tpos / (tpos + fneg)
        
    if prec + recall == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / (prec + recall)
        
    accu = 100.0 * (tpos + tneg) / (tpos + tneg + fpos + fneg)
    
    print("True Positives: %d\nFalse Positives: %d\nFalse Negatives: %d\n" % (tpos, fpos, fneg))
    print("Precision: %lf\nRecall: %lf\nAccuracy: %lf" % (prec, recall, accu))
    print("tpos,tneg,fpos,fneg:", tpos, tneg, fpos, fneg)

def main():
    from NB import classify_with_features, train 
    
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(base_dir, "reduceddata.pickle")
    
    # 检查文件是否存在
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Required file not found: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        NB.pos, NB.neg, NB.totals = pickle.load(f)
    NB.features = set(NB.pos.keys())  

    game_names = [
        "Grand Theft Auto V",
        "Tom Clancy's Rainbow Six® Siege",
        "Counter-Strike 2",
        "Dead by Daylight",
        "Call of Duty®: Black Ops III",
        "Sea of Thieves: 2024 Edition",
        "ELDEN RING",
        "Total War: WARHAMMER III",
        "Warframe",
        "Call of Duty®",
        "Apex Legends™",
        "Noita",
        "Wallpaper Engine",
        "Dragon's Dogma 2",
        "NARAKA: BLADEPOINT"
    ]
    
    for game_name in game_names:
        # 清理游戏名称以匹配文件名格式
        clean_name = game_name.replace(' ', '_').replace('®', '').replace(':', '').replace('™', '')
        paths = get_paths(clean_name)
        fscore(NB.classify_with_features, paths)  


if __name__ == '__main__':
    main()