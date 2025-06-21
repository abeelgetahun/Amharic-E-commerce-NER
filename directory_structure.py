import os

def create_project_structure(base_dir="amharic-ecommerce-ner"):
    """
    Creates the specified directory structure for the Amharic E-commerce NER project.
    """
    print(f"Creating project structure in: {os.path.abspath(base_dir)}\n")

    # Define the core directories and files
    core_dirs = [
        "config/",
        "data/raw/telegram_messages/",
        "data/raw/images/",
        "data/raw/metadata/",
        "data/processed/cleaned_text/",
        "data/processed/tokenized/",
        "data/processed/conll_format/",
        "data/labeled/validation_split/",
        "src/data_collection/",
        "src/labeling/",
        "src/models/",
        "src/utils/",
        "src/vendor_analytics/",
        "notebooks/",
        "models/checkpoints/",
        "models/fine_tuned/",
        "models/comparison_results/",
        "reports/figures/",
        "scripts/",
    ]

    # Create all directories
    for path in core_dirs:
        full_path = os.path.join(base_dir, path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

    # Create top-level placeholder files
    placeholder_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
    ]
    for file_name in placeholder_files:
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w') as f:
            if file_name == "README.md":
                f.write(f"# {base_dir}\n\nThis is the main repository for the Amharic E-commerce NER project.\n")
            elif file_name == "requirements.txt":
                f.write("# Add your Python package dependencies here (e.g., transformers, torch, pandas, telethon)\n")
            elif file_name == ".gitignore":
                f.write("""# Byte-compiled / optimized / DLL files
__pycache__/
*.pyc
*.pyd
*.pyo

# C extensions
*.so

# Distribution / packaging
.Python
env/
venv/
lib/
build/
develop-eggs/
dist/
eggs/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Editors
.vscode/
.idea/

# Jupyter Notebook files
.ipynb_checkpoints/

# Data
data/raw/ # Exclude large raw data from Git, consider LFS if needed
data/processed/ # Exclude processed data too, generate from raw
data/labeled/validation_split/ # Can be excluded if generated
models/ # Exclude large model checkpoints, use LFS or cloud storage

# Reports
reports/*.pdf # Can be excluded if generated from notebooks/scripts

# OS generated files
.DS_Store
.ipynb_checkpoints
Thumbs.db
""")
            else:
                f.write(f"# Placeholder for {file_name}\n")
        print(f"Created file: {file_path}")

    # Create __init__.py files in all src/ and config/ subdirectories
    # This marks them as Python packages
    python_package_dirs = [
        "config/",
        "src/",
        "src/data_collection/",
        "src/labeling/",
        "src/models/",
        "src/utils/",
        "src/vendor_analytics/",
    ]
    for p_dir in python_package_dirs:
        init_file_path = os.path.join(base_dir, p_dir, "__init__.py")
        if not os.path.exists(init_file_path):
            with open(init_file_path, 'w') as f:
                pass # Empty __init__.py file
            print(f"Created file: {init_file_path}")

    # Create specific files that are explicitly mentioned
    specific_files = {
        "config/settings.py": "# Project settings (API keys, paths, etc.)",
        "config/telegram_channels.yaml": "# YAML file to list Telegram channel IDs/links\nchannels:\n  - name: 'channel_1'\n    link: 'https://t.me/channel_1_link'\n  - name: 'channel_2'\n    link: 'https://t.me/channel_2_link'\n",
        "data/labeled/ner_labels.conll": "# Placeholder for manually labeled NER data in CoNLL format\n# Example:\n# Word\tLabel\n# This\tO\n# is\tO\n# a\tO\n# product\tB-Product\n# at\tO\n# 100\tB-PRICE\n# birr\tI-PRICE\n# in\tO\n# Addis\tB-LOC\n# Ababa\tI-LOC\n\n",
        "src/data_collection/telegram_scraper.py": "# Script to scrape data from Telegram channels",
        "src/data_collection/data_validator.py": "# Script to validate collected raw data",
        "src/data_collection/preprocessing.py": "# Script for initial text and image preprocessing",
        "src/labeling/conll_formatter.py": "# Script to convert raw data to CoNLL format",
        "src/labeling/entity_extractor.py": "# Script for initial entity extraction (e.g., rule-based or pre-trained model)",
        "src/labeling/label_validator.py": "# Script to validate labeled data consistency",
        "src/models/ner_trainer.py": "# Script for fine-tuning NER models (e.g., using Hugging Face Trainer)",
        "src/models/model_evaluator.py": "# Script for evaluating model performance (F1-score, precision, recall)",
        "src/models/interpretability.py": "# Script for SHAP/LIME analysis",
        "src/utils/text_processing.py": "# Utility functions for Amharic text processing (tokenization, normalization)",
        "src/utils/metrics.py": "# Utility functions for custom evaluation metrics",
        "src/utils/visualization.py": "# Utility functions for data and model visualization",
        "src/vendor_analytics/scorecard_generator.py": "# Script to generate vendor scorecards",
        "src/vendor_analytics/lending_score.py": "# Script to calculate the final lending score",
        "notebooks/01_data_exploration.ipynb": "# Jupyter notebook for initial data exploration",
        "notebooks/02_labeling_workflow.ipynb": "# Jupyter notebook for guiding through the labeling process",
        "notebooks/03_model_training.ipynb": "# Jupyter notebook for fine-tuning a single model",
        "notebooks/04_model_comparison.ipynb": "# Jupyter notebook for comparing multiple models",
        "notebooks/05_interpretability_analysis.ipynb": "# Jupyter notebook for model interpretability with SHAP/LIME",
        "notebooks/06_vendor_analytics.ipynb": "# Jupyter notebook for developing the vendor analytics engine",
        "scripts/setup_environment.py": "# Script to set up virtual environment and install dependencies",
        "scripts/run_data_collection.py": "# Script to trigger data collection",
        "scripts/run_training.py": "# Script to trigger model training",
        "scripts/generate_vendor_scorecard.py": "# Script to generate vendor scorecards",
        "reports/interim_report.pdf": "% This is a placeholder for your interim report PDF. Replace with your actual report.",
        "reports/final_report.pdf": "% This is a placeholder for your final report PDF. Replace with your actual report.",
    }

    for file_path, content in specific_files.items():
        full_path = os.path.join(base_dir, file_path)
        # Ensure parent directories exist before creating the file
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"Created file: {full_path}")

    print(f"\nProject structure '{base_dir}' created successfully!")
    print("Remember to populate 'config/telegram_channels.yaml' with actual channel links.")
    print("You can start by running 'python setup_environment.py' (after creating it) or manually installing dependencies.")

if __name__ == "__main__":
    project_name = "amharic-ecommerce-ner"
    create_project_structure(project_name)