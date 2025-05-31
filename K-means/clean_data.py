import re

def clean_data(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z1-9 `~!@#$%^&*()-_=+\[\];:'\",./?’]", '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text




