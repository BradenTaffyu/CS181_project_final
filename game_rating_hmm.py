import os
import sys
import numpy as np
import pandas as pd
import csv
import pickle
from hmmlearn import hmm
from NB import negate_sequence

class GameRatingHMM:
    def __init__(self):
        self.model_pos = None
        self.model_neg = None
        self.w2i = None
        self.n_features = None
        
    def load_model(self, model_path):
        """加载保存的HMM模型"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model_pos = model_data['model_pos']
            self.model_neg = model_data['model_neg']
            self.w2i = model_data['w2i']
            self.n_features = model_data['n_features']
    
    def predict_sentiment(self, text):
        """预测文本的情感倾向"""
        if not text:
            return 0.5  # 默认中性
            
        try:
            obs = np.array([self.w2i.get(w, self.w2i['<UNK>']) 
                           for w in negate_sequence(text)]).reshape(-1, 1)
            if len(obs) == 0:
                return 0.5
                
            pos_score = self.model_pos.score(obs)
            neg_score = self.model_neg.score(obs)
            
            # 检查分数是否有效
            if np.isnan(pos_score) or np.isnan(neg_score):
                return 0.5
                
            # 使用数值稳定的softmax计算
            scores = np.array([pos_score, neg_score])
            if np.all(np.isinf(scores)) or np.all(np.isnan(scores)):
                return 0.5
                
            # 处理无穷大的情况
            scores = np.clip(scores, -1e10, 1e10)
            scores = scores - np.max(scores)
            
            # 检查是否所有分数都是负无穷
            if np.all(np.isneginf(scores)):
                return 0.5
                
            exp_scores = np.exp(scores)
            if np.sum(exp_scores) == 0:
                return 0.5
                
            probs = exp_scores / np.sum(exp_scores)
            return float(probs[0])  # 确保返回Python float类型
            
        except Exception as e:
            print(f"预测过程中出现错误: {str(e)}")
            return 0.5

def evaluate_game_ratings(game_name, model):
    """评估游戏评分"""
    try:
        # 获取游戏评论数据
        base_dir = os.path.dirname(os.path.abspath(__file__))
        clean_name = game_name.replace(' ', '_').replace('®', '').replace(':', '').replace('™', '')
        csv_file = os.path.join(base_dir, "Data", f"{clean_name}.csv")
        
        if not os.path.exists(csv_file):
            print(f"警告: 未找到文件: {csv_file}")
            return None
            
        # 读取评论并预测情感
        ratings = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                review_text = row.get('review', '')
                sentiment = model.predict_sentiment(review_text)
                if not np.isnan(sentiment):  # 只添加有效的评分
                    ratings.append(sentiment)
        
        if not ratings:
            return None
            
        # 计算统计指标
        ratings = np.array(ratings)
        avg_rating = float(np.mean(ratings))
        std_rating = float(np.std(ratings))
        positive_ratio = float(sum(1 for r in ratings if r > 0.5) / len(ratings))
        
        return {
            'game_name': game_name,
            'average_rating': avg_rating,
            'std_rating': std_rating,
            'positive_ratio': positive_ratio,
            'total_reviews': len(ratings)
        }
    except Exception as e:
        print(f"评估游戏 {game_name} 时出现错误: {str(e)}")
        return None

def main():
    # 加载HMM模型
    print("加载HMM模型...")
    model = GameRatingHMM()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hmm_model.pickle')
    
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请先运行 HMM.py 训练并保存模型")
        return
        
    model.load_model(model_path)
    
    # 评估游戏评分
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
    
    print("\n游戏评分结果：")
    print("-" * 80)
    print(f"{'游戏名称':<40} {'平均评分':<10} {'标准差':<10} {'好评率':<10} {'评论数':<10}")
    print("-" * 80)
    
    for game_name in game_names:
        result = evaluate_game_ratings(game_name, model)
        if result:
            print(f"{result['game_name']:<40} "
                  f"{result['average_rating']:.3f}    "
                  f"{result['std_rating']:.3f}    "
                  f"{result['positive_ratio']:.3f}    "
                  f"{result['total_reviews']:<10}")

if __name__ == '__main__':
    main() 