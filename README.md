# Steam游戏评论情感分析系统

项目结构

```
.
├── HMM.py                 # HMM模型训练
├── game_rating_hmm.py     # 基于HMM的游戏评分生成系统
├── NB.py                  # 朴素贝叶斯模型训练
├── metric_modified.py     # 朴素贝叶斯的评估指标实现
├── run.ipynb             # 游戏分类的数据预处理和模型运行
├── preprocess.ipynb      # 数据预处理（无需运行）
├── countdata.pickle      # 词频统计朴素贝叶斯数据
├── reduceddata.pickle    # 精简后的朴素贝叶斯数据
├── Data/                 # 游戏评论数据
│   └── *.csv            # 各游戏的评论数据
├── pos/                  # 正面评论训练数据
├── neg/                  # 负面评论训练数据
├── test_pos/            # 正面评论测试数据
└── test_neg/            # 负面评论测试数据
```

## 使用方法

### 1. 朴素贝叶斯模型（NB）

1. 训练NB模型：

```bash
python NB.py
```

这将训练朴素贝叶斯模型并生成词频统计数据。

2. 评估NB模型：

```bash
python metric.py
```

基础评估

```bash
python metric_modified.py
```

这将使用训练好的NB模型评估游戏评论的情感分类效果。

### 2. 隐马尔可夫模型（HMM）

1. 训练HMM模型：

```bash
python HMM.py
```

这将训练HMM模型并保存到 `hmm_model.pickle`文件中。

2. 生成游戏评分：

```bash
python game_rating_hmm.py
```

这将加载训练好的模型并生成所有游戏的评分报告。

## 依赖项

- Python 3.6+
- numpy
- pandas
- hmmlearn
- scikit-learn

## 许可证

MIT License
