import asyncio
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import sys

# Import our custom modules
from src.data_collection.telegram_scraper import TelegramScraper
from src.preprocessing.amharic_processor import AmharicTextProcessor
from src.preprocessing.data_unifier import DataUnifier
from src.labeling.conll_formatter import CoNLLFormatter
from src.labeling.entity_annotator import InteractiveAnnotator

def setup_logging():
    """Setup logging configuration"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/main_execution.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

async def execute_task_1():
    """Execute Task 1: Data Ingestion and Preprocessing"""
    logger = logging.getLogger('task_1')
    logger.info("Starting Task 1: Data Ingestion and Preprocessing")
    
    # Load environment variables
    load_dotenv()
    api_id = os.getenv('TELEGRAM_API_ID')
    api_hash = os.getenv('TELEGRAM_API_HASH')
    
    if not api_id or not api_hash:
        logger.error("Telegram API credentials not found. Please set TELEGRAM_API_ID and TELEGRAM_API_HASH")
        return False
    
    try:
        # Initialize scraper
        scraper = TelegramScraper(api_id, api_hash, 'config/channels.yaml')
        await scraper.start_session()
        
        # Scrape all channels
        all_data = await scraper.scrape_all_channels()
        
        # Close session
        await scraper.close_session()
        
        # Unify and preprocess data
        logger.info("Starting data unification and preprocessing")
        unifier = DataUnifier()
        unified_df = unifier.create_unified_dataset()
        
        logger.info(f"Task 1 completed successfully. Processed {len(unified_df)} messages")
        return True
        
    except Exception as e:
        logger.error(f"Error in Task 1: {str(e)}")
        return False

def execute_task_2():
    """Execute Task 2: Data Labeling in CoNLL Format"""
    logger = logging.getLogger('task_2')
    logger.info("Starting Task 2: Data Labeling in CoNLL Format")
    
    try:
        # Load processed data
        data_path = Path("data/processed/unified_dataset.csv")
        if not data_path.exists():
            logger.error("Processed dataset not found. Please run Task 1 first.")
            return False
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} messages for labeling")
        
        # Initialize CoNLL formatter
        formatter = CoNLLFormatter()
        
        # Create initial training set with auto-labeling
        logger.info("Creating initial CoNLL training set with auto-labeling")
        conll_content = formatter.create_training_set(df, sample_size=50)
        
        # Save auto-labeled dataset
        auto_labeled_path = formatter.save_conll_dataset(conll_content, "auto_labeled_training.conll")
        logger.info(f"Auto-labeled training set saved to {auto_labeled_path}")
        
        # Interactive annotation for subset
        print("\n" + "="*60)
        print("INTERACTIVE ANNOTATION SESSION")
        print("="*60)
        
        choice = input("Do you want to start interactive annotation? (y/n): ").lower()
        if choice == 'y':
            annotator = InteractiveAnnotator()
            
            # Select high-priority messages for manual annotation
            priority_messages = df[
                df['has_price_hints'] | df['has_location_hints'] | df['has_product_hints']
            ].head(10).to_dict('records')
            
            manual_annotations = annotator.annotate_batch_interactive(priority_messages, 10)
            
            # Save manual annotations
            manual_annotations_path = Path("data/labeled/manual_annotations.json")
            with open(manual_annotations_path, 'w', encoding='utf-8') as f:
                json.dump(manual_annotations, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Manual annotations saved to {manual_annotations_path}")
        
        # Generate quality report
        generate_labeling_report(df, auto_labeled_path)
        
        logger.info("Task 2 completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in Task 2: {str(e)}")
        return False

def generate_labeling_report(df: pd.DataFrame, conll_path: Path):
    """Generate a quality report for the labeling process"""
    report = {
        'dataset_summary': {
            'total_messages': len(df),
            'amharic_messages': df['is_amharic'].sum(),
            'messages_with_entities': len(df[df['has_price_hints'] | df['has_location_hints'] | df['has_product_hints']]),
            'price_hints_found': df['has_price_hints'].sum(),
            'location_hints_found': df['has_location_hints'].sum(),
            'product_hints_found': df['has_product_hints'].sum()
        },
        'labeling_quality': {
            'auto_labeled_file': str(conll_path),
            'estimated_precision': 0.75,  # Conservative estimate for auto-labeling
            'recommended_manual_review': 30
        },
        'next_steps': [
            "Review auto-labeled data for accuracy",
            "Complete manual annotation of priority messages",
            "Validate entity boundaries and types",
            "Prepare train/validation/test splits"
        ]
    }
    
    report_path = Path("data/labeled/labeling_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nLabeling quality report saved to {report_path}")

async def main():
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger('main')
    
    print("="*60)
    print("AMHARIC E-COMMERCE NER PROJECT")
    print("Tasks 1 & 2 Execution")
    print("="*60)
    
    # Execute Task 1
    print("\n🚀 Starting Task 1: Data Ingestion and Preprocessing...")
    task1_success = await execute_task_1()
    
    if task1_success:
        print("✅ Task 1 completed successfully!")
        
        # Execute Task 2
        print("\n🚀 Starting Task 2: Data Labeling in CoNLL Format...")
        task2_success = execute_task_2()
        
        if task2_success:
            print("✅ Task 2 completed successfully!")
            print("\n🎉 Both tasks completed! Ready for model fine-tuning.")
        else:
            print("❌ Task 2 failed. Check logs for details.")
    else:
        print("❌ Task 1 failed. Cannot proceed to Task 2.")
    
    print("\n📊 Check the following directories for outputs:")
    print("- data/raw/telegram_scrapes/ - Raw scraped data")
    print("- data/processed/ - Preprocessed unified dataset")
    print("- data/labeled/ - CoNLL formatted training data")
    print("- logs/ - Execution logs")

if __name__ == "__main__":
    asyncio.run(main())