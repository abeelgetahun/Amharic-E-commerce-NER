import shap
import lime
from lime.lime_text import LimeTextExplainer
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class NERModelInterpreter:
    def __init__(self, model_path, tokenizer_path=None):
        """
        Initialize the NER Model Interpreter
        
        Args:
            model_path: Path to your fine-tuned model
            tokenizer_path: Path to tokenizer (if different from model_path)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        
        # Create NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )
        
        # Define label mappings
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
    def predict_text(self, text):
        """
        Predict entities for a given text
        
        Args:
            text: Input Amharic text
            
        Returns:
            List of predicted entities
        """
        try:
            entities = self.ner_pipeline(text)
            return entities
        except Exception as e:
            print(f"Error in prediction: {e}")
            return []
    
    def prepare_shap_explainer(self, sample_texts):
        """
        Prepare SHAP explainer for the model
        
        Args:
            sample_texts: List of sample texts for background data
        """
        def model_wrapper(texts):
            """Wrapper function for SHAP"""
            predictions = []
            for text in texts:
                try:
                    # Tokenize
                    inputs = self.tokenizer(text, return_tensors="pt", 
                                          truncation=True, padding=True)
                    
                    # Get model outputs
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        
                    # Convert to probabilities
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Average across tokens for sequence-level prediction
                    avg_probs = probs.mean(dim=1).squeeze().numpy()
                    predictions.append(avg_probs)
                    
                except Exception as e:
                    print(f"Error in model wrapper: {e}")
                    # Return zeros if error
                    num_labels = len(self.id2label)
                    predictions.append(np.zeros(num_labels))
            
            return np.array(predictions)
        
        # Create SHAP explainer
        self.shap_explainer = shap.Explainer(
            model_wrapper, 
            sample_texts[:10]  # Use subset as background
        )
        
    def explain_with_shap(self, text, max_display=10):
        """
        Explain predictions using SHAP
        
        Args:
            text: Text to explain
            max_display: Maximum number of features to display
        """
        try:
            # Get SHAP values
            shap_values = self.shap_explainer([text])
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            shap.plots.text(shap_values[0], display=False)
            plt.title(f"SHAP Explanation for Text: '{text[:50]}...'")
            plt.tight_layout()
            plt.show()
            
            return shap_values
            
        except Exception as e:
            print(f"Error in SHAP explanation: {e}")
            return None
    
    def explain_with_lime(self, text, num_features=10):
        """
        Explain predictions using LIME
        
        Args:
            text: Text to explain
            num_features: Number of features to show
        """
        try:
            def predict_proba(texts):
                """Prediction function for LIME"""
                predictions = []
                for text in texts:
                    entities = self.predict_text(text)
                    
                    # Create probability distribution based on entities found
                    probs = np.zeros(len(self.id2label))
                    if entities:
                        for entity in entities:
                            label = entity['entity_group']
                            if label in self.label2id:
                                probs[self.label2id[label]] = entity['score']
                    
                    # Normalize
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                    else:
                        probs[0] = 1.0  # Default to first class if no entities
                    
                    predictions.append(probs)
                
                return np.array(predictions)
            
            # Create LIME explainer
            explainer = LimeTextExplainer(
                class_names=list(self.id2label.values()),
                mode='classification'
            )
            
            # Generate explanation
            explanation = explainer.explain_instance(
                text, 
                predict_proba, 
                num_features=num_features
            )
            
            # Show explanation
            explanation.show_in_notebook(text=True)
            
            return explanation
            
        except Exception as e:
            print(f"Error in LIME explanation: {e}")
            return None
    
    def analyze_difficult_cases(self, test_texts, true_labels=None):
        """
        Analyze difficult cases where model struggles
        
        Args:
            test_texts: List of test texts
            true_labels: Optional true labels for comparison
        """
        difficult_cases = []
        
        for i, text in enumerate(test_texts):
            try:
                # Get predictions
                entities = self.predict_text(text)
                
                # Analyze prediction confidence
                low_confidence_entities = [
                    e for e in entities if e['score'] < 0.7
                ]
                
                # Check for overlapping entities
                overlapping = self._check_overlapping_entities(entities)
                
                # Check for ambiguous text patterns
                ambiguous = self._check_ambiguous_patterns(text)
                
                if low_confidence_entities or overlapping or ambiguous:
                    case = {
                        'text': text,
                        'entities': entities,
                        'low_confidence': low_confidence_entities,
                        'overlapping': overlapping,
                        'ambiguous_patterns': ambiguous,
                        'index': i
                    }
                    difficult_cases.append(case)
                    
            except Exception as e:
                print(f"Error analyzing text {i}: {e}")
        
        return difficult_cases
    
    def _check_overlapping_entities(self, entities):
        """Check for overlapping entity spans"""
        overlapping = []
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                ent1, ent2 = entities[i], entities[j]
                if (ent1['start'] < ent2['end'] and ent2['start'] < ent1['end']):
                    overlapping.append((ent1, ent2))
        return overlapping
    
    def _check_ambiguous_patterns(self, text):
        """Check for ambiguous text patterns"""
        ambiguous_patterns = []
        
        # Check for mixed languages
        english_chars = sum(1 for c in text if ord(c) < 128)
        amharic_chars = sum(1 for c in text if ord(c) >= 4608 and ord(c) <= 4991)
        
        if english_chars > 0 and amharic_chars > 0:
            ambiguous_patterns.append("Mixed language text")
        
        # Check for multiple price patterns
        price_indicators = ['ብር', 'birr', 'ETB', '₹']
        price_count = sum(text.lower().count(indicator) for indicator in price_indicators)
        if price_count > 1:
            ambiguous_patterns.append("Multiple price indicators")
        
        return ambiguous_patterns
    
    def generate_interpretability_report(self, sample_texts, output_path="interpretability_report.html"):
        """
        Generate comprehensive interpretability report
        
        Args:
            sample_texts: List of sample texts to analyze
            output_path: Path to save the HTML report
        """
        report_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>NER Model Interpretability Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin: 30px 0; }
                .entity { padding: 5px; margin: 5px; border-radius: 3px; }
                .B-Product { background-color: #FFE6E6; }
                .B-PRICE { background-color: #E6F3FF; }
                .B-LOC { background-color: #E6FFE6; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>NER Model Interpretability Report</h1>
        """
        
        # Analyze sample texts
        for i, text in enumerate(sample_texts[:5]):  # Analyze first 5 texts
            entities = self.predict_text(text)
            
            report_html += f"""
            <div class="section">
                <h3>Sample {i+1}: {text[:100]}...</h3>
                <h4>Predicted Entities:</h4>
                <ul>
            """
            
            for entity in entities:
                report_html += f"""
                    <li class="entity {entity['entity_group']}">
                        <strong>{entity['entity_group']}</strong>: {entity['word']} 
                        (Confidence: {entity['score']:.3f})
                    </li>
                """
            
            report_html += "</ul></div>"
        
        # Analyze difficult cases
        difficult_cases = self.analyze_difficult_cases(sample_texts)
        
        report_html += f"""
        <div class="section">
            <h2>Difficult Cases Analysis</h2>
            <p>Found {len(difficult_cases)} difficult cases out of {len(sample_texts)} samples.</p>
            <table>
                <tr>
                    <th>Text</th>
                    <th>Issues</th>
                    <th>Low Confidence Entities</th>
                </tr>
        """
        
        for case in difficult_cases[:10]:  # Show first 10 difficult cases
            issues = []
            if case['low_confidence']:
                issues.append(f"{len(case['low_confidence'])} low confidence entities")
            if case['overlapping']:
                issues.append(f"{len(case['overlapping'])} overlapping entities")
            if case['ambiguous_patterns']:
                issues.extend(case['ambiguous_patterns'])
            
            low_conf_text = ", ".join([
                f"{e['entity_group']}:{e['score']:.2f}" 
                for e in case['low_confidence']
            ])
            
            report_html += f"""
                <tr>
                    <td>{case['text'][:100]}...</td>
                    <td>{"; ".join(issues)}</td>
                    <td>{low_conf_text}</td>
                </tr>
            """
        
        report_html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Recommendations for Model Improvement</h2>
            <ul>
                <li>Increase training data for entities with low confidence scores</li>
                <li>Add more diverse examples for ambiguous patterns</li>
                <li>Implement post-processing to handle overlapping entities</li>
                <li>Consider ensemble methods for better confidence calibration</li>
            </ul>
        </div>
        
        </body>
        </html>
        """
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        print(f"Interpretability report saved to {output_path}")

# Usage example
def run_interpretability_analysis():
    """
    Main function to run interpretability analysis
    """
    # Initialize interpreter with your fine-tuned model
    interpreter = NERModelInterpreter(
        model_path="path/to/your/fine-tuned-model",  # Update this path
        tokenizer_path="path/to/your/tokenizer"      # Update this path
    )
    
    # Sample Amharic texts for analysis
    sample_texts = [
        "የመሸሻ ቦርሳ ዋጋ 500 ብር በቦሌ አካባቢ",
        "የልጆች ጫማ በ 250 ብር በመርካቶ ይሸጣል",
        "ስልክ cover በ 50 ብር Addis Ababa ውስጥ",
        # Add more sample texts from your dataset
    ]
    
    # Prepare SHAP explainer
    print("Preparing SHAP explainer...")
    interpreter.prepare_shap_explainer(sample_texts)
    
    # Explain individual predictions
    print("\nExplaining with SHAP...")
    for text in sample_texts[:2]:
        interpreter.explain_with_shap(text)
    
    print("\nExplaining with LIME...")
    for text in sample_texts[:2]:
        interpreter.explain_with_lime(text)
    
    # Analyze difficult cases
    print("\nAnalyzing difficult cases...")
    difficult_cases = interpreter.analyze_difficult_cases(sample_texts)
    print(f"Found {len(difficult_cases)} difficult cases")
    
    # Generate comprehensive report
    print("\nGenerating interpretability report...")
    interpreter.generate_interpretability_report(sample_texts)
    
    return interpreter

if __name__ == "__main__":
    interpreter = run_interpretability_analysis()