# Script for initial text and image preprocessing
import re
import pandas as pd

class AmharicTextPreprocessor:
    def __init__(self):
        self.amharic_pattern = re.compile(r'[\u1200-\u137F]+')
        
    def clean_text(self, text):
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def tokenize_amharic(self, text):
        # Simple tokenization for Amharic
        tokens = text.split()
        return [token for token in tokens if token.strip()]
        
    def preprocess_dataframe(self, df):
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_amharic)
        df['token_count'] = df['tokens'].apply(len)
        return df