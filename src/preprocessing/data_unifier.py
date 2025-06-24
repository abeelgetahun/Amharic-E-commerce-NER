import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
from preprocessing.amharic_processor import AmharicTextProcessor

class DataUnifier:
    def __init__(self, raw_data_path: str = "data/raw/telegram_scrapes"):
        self.raw_data_path = Path(raw_data_path)
        # Always resolve path from project root, not from notebook location
        project_root = Path(__file__).resolve().parent.parent.parent
        self.processed_data_path = project_root / "data" / "processed"
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger('data_unifier')
        self.text_processor = AmharicTextProcessor()
    
    def load_all_channel_data(self) -> pd.DataFrame:
        """Load and combine data from all scraped channels"""
        all_messages = []
        
        for channel_dir in self.raw_data_path.iterdir():
            if channel_dir.is_dir():
                json_file = channel_dir / f"{channel_dir.name}_messages.json"
                if json_file.exists():
                    with open(json_file, 'r', encoding='utf-8') as f:
                        channel_data = json.load(f)
                        all_messages.extend(channel_data)
                        self.logger.info(f"Loaded {len(channel_data)} messages from {channel_dir.name}")
        
        df = pd.DataFrame(all_messages)
        self.logger.info(f"Combined dataset contains {len(df)} total messages")
        return df
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the combined dataset"""
        processed_df = df.copy()
        
        # Clean text data
        processed_df['cleaned_text'] = processed_df['text'].apply(self.text_processor.clean_text)
        
        # Filter for Amharic-dominant messages
        processed_df['is_amharic'] = processed_df['cleaned_text'].apply(self.text_processor.is_amharic_dominant)
        
        # Extract entity hints
        processed_df['entity_hints'] = processed_df['cleaned_text'].apply(self.text_processor.extract_entities_hints)
        
        # Tokenize text
        processed_df['tokens'] = processed_df['cleaned_text'].apply(self.text_processor.tokenize_amharic)
        processed_df['token_count'] = processed_df['tokens'].apply(len)
        
        # Add processing metadata
        processed_df['processed_at'] = datetime.now().isoformat()
        processed_df['has_price_hints'] = processed_df['entity_hints'].apply(lambda x: len(x['price_hints']) > 0)
        processed_df['has_location_hints'] = processed_df['entity_hints'].apply(lambda x: len(x['location_hints']) > 0)
        processed_df['has_product_hints'] = processed_df['entity_hints'].apply(lambda x: len(x['product_hints']) > 0)
        
        # Filter out very short or empty messages
        processed_df = processed_df[processed_df['token_count'] >= 3]
        
        self.logger.info(f"After preprocessing: {len(processed_df)} messages remain")
        return processed_df
    
    def create_unified_dataset(self) -> pd.DataFrame:
        """Create the final unified dataset"""
        # Load all data
        raw_df = self.load_all_channel_data()
        
        # Preprocess
        processed_df = self.preprocess_dataset(raw_df)
        
        # Save processed dataset
        self.save_processed_data(processed_df)
        
        return processed_df
    
    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data in multiple formats"""
        # Save as CSV
        csv_path = self.processed_data_path / "unified_dataset.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save as JSON for complex nested data
        json_path = self.processed_data_path / "unified_dataset.json"
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        
        # Save summary statistics
        summary = self.generate_dataset_summary(df)
        summary_path = self.processed_data_path / "dataset_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Processed data saved to {self.processed_data_path}")
    
    def generate_dataset_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive dataset summary"""
        # Convert dates to string to ensure JSON serializability
        earliest = str(df['date'].min()) if not pd.isnull(df['date'].min()) else None
        latest = str(df['date'].max()) if not pd.isnull(df['date'].max()) else None

        summary = {
            'total_messages': int(len(df)),
            'channels': [str(ch) for ch in df['channel_username'].unique().tolist()],
            'messages_per_channel': {str(k): int(v) for k, v in df['channel_username'].value_counts().to_dict().items()},
            'date_range': {
                'earliest': earliest,
                'latest': latest
            },
            'amharic_messages': int(df['is_amharic'].sum()),
            'messages_with_media': int(df['has_media'].sum()),
            'messages_with_price_hints': int(df['has_price_hints'].sum()),
            'messages_with_location_hints': int(df['has_location_hints'].sum()),
            'messages_with_product_hints': int(df['has_product_hints'].sum()),
            'avg_token_count': float(df['token_count'].mean()),
            'avg_views_per_message': float(df['views'].mean()),
            'total_views': int(df['views'].sum())
        }
        return summary