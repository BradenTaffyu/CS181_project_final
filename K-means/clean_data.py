import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

def clean_data(text):
    # 清理步骤1：转换为小写并移除非字母数字字符
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", '', text)  # 移除非字母字符（保留空格）
    
    # 清理步骤2：分词并过滤停用词
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # 清理步骤3：合并空白字符并修剪
    cleaned_text = ' '.join(filtered_words)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text



