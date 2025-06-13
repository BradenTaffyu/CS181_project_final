import os
import numpy as np
from HMM import HMM
from NB import negate_sequence

def test_hmm_model():
    """测试HMM模型的训练和预测"""
    print("开始测试HMM模型...")
    
    # 初始化HMM模型
    model = HMM()
    
    # 测试数据路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pos_dir = os.path.join(base_dir, "pos")
    neg_dir = os.path.join(base_dir, "neg")
    
    # 训练模型
    print("\n训练HMM模型...")
    model.train(pos_dir, neg_dir)
    
    # 保存模型
    model_path = os.path.join(base_dir, "hmm_model.pickle")
    model.save_model(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 测试预测
    print("\n测试预测功能...")
    test_texts = [
        "This game is amazing! I love it!",
        "Terrible game, waste of money.",
        "The graphics are good but the gameplay is boring.",
        "Best game I've ever played!",
        "Not worth the price at all."
    ]
    
    print("\n预测结果：")
    print("-" * 80)
    print(f"{'测试文本':<40} {'情感得分':<10}")
    print("-" * 80)
    
    for text in test_texts:
        score = model.predict_sentiment(text)
        print(f"{text[:40]:<40} {score:.3f}")
    
    # 测试模型加载
    print("\n测试模型加载...")
    new_model = HMM()
    new_model.load_model(model_path)
    
    # 验证加载的模型
    print("\n验证加载的模型...")
    for text in test_texts:
        original_score = model.predict_sentiment(text)
        loaded_score = new_model.predict_sentiment(text)
        print(f"文本: {text[:30]}...")
        print(f"原始模型得分: {original_score:.3f}")
        print(f"加载模型得分: {loaded_score:.3f}")
        print(f"得分差异: {abs(original_score - loaded_score):.6f}")
        print("-" * 40)

if __name__ == "__main__":
    test_hmm_model() 