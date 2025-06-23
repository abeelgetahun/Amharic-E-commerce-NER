# Script to convert raw data to CoNLL format
import pandas as pd
import json
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

class CoNLLFormatter:
    def __init__(self):
        self.logger = logging.getLogger('conll_formatter')
        
        # Entity type mappings
        self.entity_types = {
            'PRODUCT': ['B-Product', 'I-Product'],
            'LOCATION': ['B-LOC', 'I-LOC'], 
            'PRICE': ['B-PRICE', 'I-PRICE']
        }
        
        # Common product keywords in Amharic
        self.product_keywords = [
            'ሸሚዝ', 'ቱንጃ', 'ጃኬት', 'ቦርሳ', 'ጫማ', 'ወይን', 'ቡና', 'ሻሁ', 'ሱራ',
            'ዱቄት', 'በርበሬ', 'ዘይት', 'ስንዴ', 'ባቄላ', 'ሽሮ', 'ሳር', 'መጽሀፍ', 'ብዕር',
            'ኮምፒውተር', 'ስልክ', 'ካሜራ', 'ቴሌቪዥን', 'ሰዓት', 'መነፅር', 'መዳፊት'
        ]
        
        # Location keywords
        self.location_keywords = [
            'አዲስ', 'አበባ', 'ቦሌ', 'ጣና', 'መገናኛ', 'ሀዋሳ', 'ባህር ዳር', 'መቀሌ', 
            'ጎንደር', 'ደሴ', 'አርባ ምንች', 'ጅማ', 'አሶሳ', 'ጋምቤላ'
        ]
        
        # Price indicators
        self.price_indicators = [
            'ዋጋ', 'ብር', 'birr', 'price', 'በ', 'ዶላር', 'dollar'
        ]
    
    def tokenize_for_labeling(self, text: str) -> List[str]:
        """Tokenize text specifically for labeling purposes"""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split while preserving punctuation as separate tokens
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in '.,!?;:()[]{}።፣፤፥፦':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return [token for token in tokens if token.strip()]
    
    def auto_label_entities(self, tokens: List[str]) -> List[str]:
        """Automatically generate initial entity labels"""
        labels = ['O'] * len(tokens)
        i = 0
        
        while i < len(tokens):
            token = tokens[i].lower()
            
            # Check for price patterns
            if self._is_price_indicator(token):
                price_span = self._find_price_span(tokens, i)
                if price_span:
                    start, end = price_span
                    labels[start] = 'B-PRICE'
                    for j in range(start + 1, end + 1):
                        if j < len(labels):
                            labels[j] = 'I-PRICE'
                    i = end + 1
                    continue
            
            # Check for location patterns
            if self._is_location_indicator(token):
                location_span = self._find_location_span(tokens, i)
                if location_span:
                    start, end = location_span
                    labels[start] = 'B-LOC'
                    for j in range(start + 1, end + 1):
                        if j < len(labels):
                            labels[j] = 'I-LOC'
                    i = end + 1
                    continue
            
            # Check for product patterns
            if self._is_product_indicator(token):
                product_span = self._find_product_span(tokens, i)
                if product_span:
                    start, end = product_span
                    labels[start] = 'B-Product'
                    for j in range(start + 1, end + 1):
                        if j < len(labels):
                            labels[j] = 'I-Product'
                    i = end + 1
                    continue
            
            i += 1
        
        return labels
    
    def _is_price_indicator(self, token: str) -> bool:
        """Check if token indicates a price"""
        token_lower = token.lower()
        
        # Direct price indicators
        if any(indicator in token_lower for indicator in self.price_indicators):
            return True
        
        # Number followed by currency
        if re.match(r'\d+', token):
            return True
        
        return False
    
    def _find_price_span(self, tokens: List[str], start_idx: int) -> Optional[Tuple[int, int]]:
        """Find the span of a price entity"""
        end_idx = start_idx
        
        # Look for number + currency pattern
        for i in range(start_idx, min(start_idx + 5, len(tokens))):
            token = tokens[i].lower()
            if any(indicator in token for indicator in self.price_indicators) or re.search(r'\d', token):
                end_idx = i
            else:
                break
        
        return (start_idx, end_idx) if end_idx > start_idx or self._is_price_indicator(tokens[start_idx]) else None
    
    def _is_location_indicator(self, token: str) -> bool:
        """Check if token indicates a location"""
        return any(loc in token.lower() for loc in self.location_keywords)
    
    def _find_location_span(self, tokens: List[str], start_idx: int) -> Optional[Tuple[int, int]]:
        """Find the span of a location entity"""
        end_idx = start_idx
        
        # Look for multi-word locations like "አዲስ አበባ"
        for i in range(start_idx, min(start_idx + 3, len(tokens))):
            if self._is_location_indicator(tokens[i]):
                end_idx = i
            else:
                break
        
        return (start_idx, end_idx) if end_idx >= start_idx else None
    
    def _is_product_indicator(self, token: str) -> bool:
        """Check if token indicates a product"""
        return any(product in token.lower() for product in self.product_keywords)
    
    def _find_product_span(self, tokens: List[str], start_idx: int) -> Optional[Tuple[int, int]]:
        """Find the span of a product entity"""
        end_idx = start_idx
        
        # Look for product names (usually 1-3 words)
        for i in range(start_idx, min(start_idx + 3, len(tokens))):
            token = tokens[i]
            if (self._is_product_indicator(token) or 
                (i > start_idx and not token.lower() in ['እና', 'ወይም', 'ላይ', 'ውስጥ']) and
                not re.match(r'^[^\u1200-\u137F]+$', token)):  # Not purely non-Amharic
                end_idx = i
            else:
                break
        
        return (start_idx, end_idx) if end_idx >= start_idx else None
    
    def format_message_to_conll(self, text: str, message_id: str = "") -> str:
        """Convert a single message to CoNLL format"""
        tokens = self.tokenize_for_labeling(text)
        labels = self.auto_label_entities(tokens)
        
        conll_lines = []
        if message_id:
            conll_lines.append(f"# Message ID: {message_id}")
        
        for token, label in zip(tokens, labels):
            conll_lines.append(f"{token}\t{label}")
        
        conll_lines.append("")  # Empty line to separate messages
        
        return "\n".join(conll_lines)
    
    def create_training_set(self, df: pd.DataFrame, sample_size: int = 50) -> str:
        """Create a training set in CoNLL format from DataFrame"""
        # Sample messages with good entity hints
        sampled_df = self._smart_sample_messages(df, sample_size)
        
        conll_content = []
        conll_content.append("# Amharic E-commerce NER Training Data")
        conll_content.append("# Format: TOKEN\tLABEL")
        conll_content.append("# Entity types: B-Product, I-Product, B-LOC, I-LOC, B-PRICE, I-PRICE, O")
        conll_content.append("")
        
        for idx, row in sampled_df.iterrows():
            message_conll = self.format_message_to_conll(
                row['cleaned_text'], 
                str(row['message_id'])
            )
            conll_content.append(message_conll)
        
        return "\n".join(conll_content)
    
    def _smart_sample_messages(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Intelligently sample messages for labeling"""
        # Prioritize messages with entity hints
        priority_messages = df[
            df['has_price_hints'] | df['has_location_hints'] | df['has_product_hints']
        ].copy()
        
        # Add diversity score
        priority_messages['diversity_score'] = (
            priority_messages['has_price_hints'].astype(int) +
            priority_messages['has_location_hints'].astype(int) +
            priority_messages['has_product_hints'].astype(int) +
            (priority_messages['token_count'] / priority_messages['token_count'].max())
        )
        
        # Sample based on diversity and entity presence
        if len(priority_messages) >= sample_size:
            sampled = priority_messages.nlargest(sample_size, 'diversity_score')
        else:
            # If not enough priority messages, fill with random samples
            remaining_needed = sample_size - len(priority_messages)
            other_messages = df[~df.index.isin(priority_messages.index)]
            additional_samples = other_messages.sample(min(remaining_needed, len(other_messages)))
            sampled = pd.concat([priority_messages, additional_samples])
        
        return sampled.reset_index(drop=True)
    
    def save_conll_dataset(self, conll_content: str, filename: str = "training_data.conll"):
        """Save CoNLL formatted data to file"""
        output_path = Path("data/labeled/conll_format")
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(conll_content)
        
        self.logger.info(f"CoNLL dataset saved to {file_path}")
        return file_path