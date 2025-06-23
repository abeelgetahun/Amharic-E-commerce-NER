import re
import string
import pandas as pd
from typing import List, Dict, Tuple
import logging

class AmharicTextProcessor:
    def __init__(self):
        self.logger = logging.getLogger('amharic_processor')
        
        # Amharic Unicode ranges
        self.amharic_range = r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]'
        
        # Common Amharic stopwords (basic set)
        self.amharic_stopwords = {
            'እና', 'ወይም', 'ግን', 'ነው', 'ናት', 'ናቸው', 'አለ', 'አላት', 'ዘንድ', 
            'ለ', 'በ', 'ከ', 'እስከ', 'ወደ', 'ላይ', 'ስር', 'ውስጥ', 'ፊት', 'ኋላ'
        }
        
        # Price-related patterns
        self.price_patterns = [
            r'ዋጋ\s*[:፦]?\s*\d+',
            r'\d+\s*ብር',
            r'በ\s*\d+\s*ብር',
            r'ብር\s*\d+',
            r'\d+\s*birr',
            r'price\s*[:፦]?\s*\d+'
        ]
        
        # Location indicators
        self.location_indicators = [
            'አዲስ አበባ', 'አዲስ', 'አበባ', 'መገናኛ', 'ቦሌ', 'ጣና', 'ወሎ ሰፈር',
            'ቢሸፍቱ', 'ጋሪ', 'ጉልፍ', 'ሀዋሳ', 'ባህር ዳር', 'መቀሌ', 'ጎንደር'
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Amharic text"""
        if not text or pd.isna(text):
            return ""
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Normalize Amharic punctuation
        text = text.replace('፡', ':')
        text = text.replace('።', '.')
        text = text.replace('፣', ',')
        text = text.replace('፤', ';')
        text = text.replace('፥', ':')
        text = text.replace('፦', ':')
        
        # Remove emoji patterns (basic)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)
        
        return text.strip()
    
    def tokenize_amharic(self, text: str) -> List[str]:
        """Tokenize Amharic text"""
        if not text:
            return []
        
        # Split by whitespace and punctuation while preserving Amharic characters
        tokens = re.findall(r'\S+', text)
        
        # Further split tokens that contain mixed scripts
        processed_tokens = []
        for token in tokens:
            # If token contains both Amharic and non-Amharic, try to split
            if re.search(self.amharic_range, token) and re.search(r'[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\s]', token):
                # Split mixed tokens
                subtokens = re.findall(r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]+|[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\s]+', token)
                processed_tokens.extend([t for t in subtokens if t.strip()])
            else:
                if token.strip():
                    processed_tokens.append(token)
        
        return processed_tokens
    
    def extract_entities_hints(self, text: str) -> Dict[str, List[str]]:
        """Extract potential entity hints from text"""
        entities = {
            'price_hints': [],
            'location_hints': [],
            'product_hints': []
        }
        
        # Extract price hints
        for pattern in self.price_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities['price_hints'].append(match.group())
        
        # Extract location hints
        for location in self.location_indicators:
            if location.lower() in text.lower():
                entities['location_hints'].append(location)
        
        # Simple product hints (nouns after common product indicators)
        product_indicators = ['ዕቃ', 'ምርት', 'አይነት', 'ሸቀጣሸቀጥ']
        for indicator in product_indicators:
            pattern = rf'{indicator}\s+([^\s]+(?:\s+[^\s]+)?)'
            matches = re.finditer(pattern, text)
            for match in matches:
                entities['product_hints'].append(match.group(1))
        
        return entities
    
    def is_amharic_dominant(self, text: str) -> bool:
        """Check if text is predominantly Amharic"""
        if not text:
            return False
        
        amharic_chars = len(re.findall(self.amharic_range, text))
        total_chars = len(re.sub(r'\s+', '', text))
        
        if total_chars == 0:
            return False
        
        return (amharic_chars / total_chars) > 0.5
    
    def normalize_numbers(self, text: str) -> str:
        """Normalize number representations"""
        # Convert Amharic numbers to Arabic numerals if needed
        amharic_to_arabic = {
            '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
            '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10'
        }
        
        for amharic, arabic in amharic_to_arabic.items():
            text = text.replace(amharic, arabic)
        
        return text