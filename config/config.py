"""
Configuration file for EthioMart NER Project
Update these paths according to your project structure
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data paths (from Task 1)
TELEGRAM_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telegram_data.csv")
LABELED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "labeled_data.conll")

# Model paths (from Tasks 3-4)
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tokenizer")

# Output paths
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Create directories if they don't exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Interpretability settings
INTERPRETABILITY_CONFIG = {
    "max_samples_for_shap": 10,
    "max_samples_for_lime": 5,
    "num_features_lime": 10
}

# Vendor scoring weights
VENDOR_SCORING_WEIGHTS = {
    'avg_views_per_post': 0.3,
    'posting_frequency': 0.2,
    'avg_price_point': 0.2,
    'product_diversity': 0.1,
    'engagement_trend': 0.1,
    'consistency_score': 0.1
}