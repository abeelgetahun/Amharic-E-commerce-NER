import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from collections import defaultdict
import json

class VendorAnalyticsEngine:
    def __init__(self, telegram_data_path, ner_model=None):
        """
        Initialize the Vendor Analytics Engine
        
        Args:
            telegram_data_path: Path to your scraped Telegram data
            ner_model: Your trained NER model for entity extraction
        """
        self.data = pd.read_csv(telegram_data_path)
        self.ner_model = ner_model
        self.vendor_metrics = {}
        
        # Ensure required columns exist
        required_columns = ['channel_name', 'message', 'timestamp', 'views']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Convert timestamp to datetime
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
    def extract_entities_from_messages(self):
        """
        Extract entities from messages using the NER model
        """
        print("Extracting entities from messages...")
        
        entities_data = []
        
        for idx, row in self.data.iterrows():
            message = row['message']
            channel = row['channel_name']
            
            if self.ner_model:
                try:
                    # Use your NER model to extract entities
                    entities = self.ner_model.predict_text(message)
                    
                    # Process entities
                    products = []
                    prices = []
                    locations = []
                    
                    for entity in entities:
                        if entity['entity_group'] in ['B-Product', 'I-Product']:
                            products.append(entity['word'])
                        elif entity['entity_group'] in ['B-PRICE', 'I-PRICE']:
                            prices.append(entity['word'])
                        elif entity['entity_group'] in ['B-LOC', 'I-LOC']:
                            locations.append(entity['word'])
                    
                    entities_data.append({
                        'message_id': idx,
                        'channel_name': channel,
                        'products': products,
                        'prices': prices,
                        'locations': locations,
                        'message': message,
                        'timestamp': row['timestamp'],
                        'views': row['views']
                    })
                    
                except Exception as e:
                    print(f"Error processing message {idx}: {e}")
                    
            else:
                # Fallback: Use simple regex patterns for entity extraction
                products = self._extract_products_regex(message)
                prices = self._extract_prices_regex(message)
                locations = self._extract_locations_regex(message)
                
                entities_data.append({
                    'message_id': idx,
                    'channel_name': channel,
                    'products': products,
                    'prices': prices,
                    'locations': locations,
                    'message': message,
                    'timestamp': row['timestamp'],
                    'views': row['views']
                })
        
        self.entities_df = pd.DataFrame(entities_data)
        return self.entities_df
    
    def _extract_products_regex(self, text):
        """Fallback regex-based product extraction"""
        # Simple patterns for Amharic products
        product_patterns = [
            r'የ[\w\s]+(?:ቦርሳ|ጫማ|ሸሚዝ|ቀሚስ|ሱሪ)',
            r'[\w\s]*(?:cover|case|phone|mobile)',
            r'[\w\s]*(?:ልጆች|ሴቶች|ወንዶች)[\w\s]*'
        ]
        
        products = []
        for pattern in product_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            products.extend(matches)
        
        return products
    
    def _extract_prices_regex(self, text):
        """Fallback regex-based price extraction"""
        price_patterns = [
            r'\d+\s*ብር',
            r'\d+\s*birr',
            r'\d+\s*ETB',
            r'በ\s*\d+',
            r'ዋጋ\s*\d+'
        ]
        
        prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            prices.extend(matches)
        
        return prices
    
    def _extract_locations_regex(self, text):
        """Fallback regex-based location extraction"""
        locations = [
            'አዲስ አበባ', 'Addis Ababa', 'ቦሌ', 'Bole', 'መርካቶ', 'Merkato',
            'ፒያሳ', 'Piassa', '4 ኪሎ', '6 ኪሎ', 'ካዛንቺስ', 'Kazanchis'
        ]
        
        found_locations = []
        for location in locations:
            if location.lower() in text.lower():
                found_locations.append(location)
        
        return found_locations
    
    def calculate_vendor_metrics(self):
        """
        Calculate key performance metrics for each vendor
        """
        print("Calculating vendor metrics...")
        
        if not hasattr(self, 'entities_df'):
            self.extract_entities_from_messages()
        
        vendor_metrics = {}
        
        for channel in self.entities_df['channel_name'].unique():
            channel_data = self.entities_df[
                self.entities_df['channel_name'] == channel
            ].copy()
            
            # Activity & Consistency Metrics
            posting_frequency = self._calculate_posting_frequency(channel_data)
            
            # Market Reach & Engagement Metrics
            avg_views_per_post = channel_data['views'].mean()
            top_performing_post = self._get_top_performing_post(channel_data)
            
            # Business Profile Metrics
            avg_price_point = self._calculate_average_price(channel_data)
            product_diversity = self._calculate_product_diversity(channel_data)
            
            # Additional Metrics
            engagement_trend = self._calculate_engagement_trend(channel_data)
            consistency_score = self._calculate_consistency_score(channel_data)
            
            vendor_metrics[channel] = {
                'posting_frequency': posting_frequency,
                'avg_views_per_post': avg_views_per_post,
                'top_performing_post': top_performing_post,
                'avg_price_point': avg_price_point,
                'product_diversity': product_diversity,
                'engagement_trend': engagement_trend,
                'consistency_score': consistency_score,
                'total_posts': len(channel_data),
                'total_views': channel_data['views'].sum()
            }
        
        self.vendor_metrics = vendor_metrics
        return vendor_metrics
    
    def _calculate_posting_frequency(self, channel_data):
        """Calculate average posts per week"""
        if len(channel_data) == 0:
            return 0
        
        date_range = (
            channel_data['timestamp'].max() - 
            channel_data['timestamp'].min()
        ).days
        
        if date_range == 0:
            return len(channel_data)
        
        weeks = max(date_range / 7, 1)
        return len(channel_data) / weeks
    
    def _get_top_performing_post(self, channel_data):
        """Get the post with highest views"""
        if len(channel_data) == 0:
            return None
        
        top_post = channel_data.loc[channel_data['views'].idxmax()]
        
        return {
            'views': top_post['views'],
            'message': top_post['message'][:100] + "...",
            'products': top_post['products'],
            'prices': top_post['prices'],
            'timestamp': top_post['timestamp']
        }
    
    def _calculate_average_price(self, channel_data):
        """Calculate average price point from extracted prices"""
        all_prices = []
        
        for prices in channel_data['prices']:
            for price_text in prices:
                # Extract numeric value from price text
                numbers = re.findall(r'\d+', price_text)
                if numbers:
                    all_prices.append(float(numbers[0]))
        
        return np.mean(all_prices) if all_prices else 0
    
    def _calculate_product_diversity(self, channel_data):
        """Calculate product diversity score"""
        all_products = []
        for products in channel_data['products']:
            all_products.extend(products)
        
        unique_products = len(set(all_products))
        total_products = len(all_products)
        
        return unique_products / max(total_products, 1)
    
    def _calculate_engagement_trend(self, channel_data):
        """Calculate engagement trend (positive/negative/stable)"""
        if len(channel_data) < 2:
            return 0
        
        # Sort by timestamp
        sorted_data = channel_data.sort_values('timestamp')
        
        # Calculate trend using linear regression on views
        x = np.arange(len(sorted_data))
        y = sorted_data['views'].values
        
        # Simple trend calculation
        trend = np.polyfit(x, y, 1)[0]
        return trend
    
    def _calculate_consistency_score(self, channel_data):
        """Calculate posting consistency score"""
        if len(channel_data) < 2:
            return 0
        
        # Calculate variance in posting intervals
        sorted_data = channel_data.sort_values('timestamp')
        intervals = sorted_data['timestamp'].diff().dt.days.dropna()
        
        if len(intervals) == 0:
            return 0
        
        # Lower variance = higher consistency
        variance = intervals.var()
        consistency = 1 / (1 + variance) if variance > 0 else 1
        
        return consistency
    
    def calculate_lending_scores(self, weights=None):
        """
        Calculate lending scores for all vendors
        
        Args:
            weights: Dictionary of metric weights for scoring
        """
        if weights is None:
            weights = {
                'avg_views_per_post': 0.3,
                'posting_frequency': 0.2,
                'avg_price_point': 0.2,
                'product_diversity': 0.1,
                'engagement_trend': 0.1,
                'consistency_score': 0.1
            }
        
        if not self.vendor_metrics:
            self.calculate_vendor_metrics()
        
        # Normalize metrics for scoring
        all_metrics = pd.DataFrame(self.vendor_metrics).T
        
        # Normalize each metric to 0-1 scale
        normalized_metrics = {}
        for metric in weights.keys():
            if metric in all_metrics.columns:
                values = all_metrics[metric].values
                min_val, max_val = values.min(), values.max()
                if max_val > min_val:
                    normalized_values = (values - min_val) / (max_val - min_val)
                else:
                    normalized_values = np.ones_like(values)
                normalized_metrics[metric] = normalized_values
        
        # Calculate lending scores
        lending_scores = {}
        for i, vendor in enumerate(all_metrics.index):
            score = sum(
                normalized_metrics[metric][i] * weight
                for metric, weight in weights.items()
                if metric in normalized_metrics
            )
            lending_scores[vendor] = score
        
        # Add lending scores to vendor metrics
        for vendor, score in lending_scores.items():
            self.vendor_metrics[vendor]['lending_score'] = score
        
        return lending_scores
    
    def create_vendor_scorecard(self, output_path="vendor_scorecard.html"):
        """
        Create a comprehensive vendor scorecard
        """
        if not self.vendor_metrics:
            self.calculate_vendor_metrics()
            self.calculate_lending_scores()
        
        # Create summary DataFrame
        scorecard_data = []
        for vendor, metrics in self.vendor_metrics.items():
            scorecard_data.append({
                'Vendor Channel': vendor,
                'Avg. Views/Post': f"{metrics['avg_views_per_post']:.0f}",
                'Posts/Week': f"{metrics['posting_frequency']:.1f}",
                'Avg. Price (ETB)': f"{metrics['avg_price_point']:.0f}",
                'Product Diversity': f"{metrics['product_diversity']:.2f}",
                'Consistency Score': f"{metrics['consistency_score']:.2f}",
                'Lending Score': f"{metrics['lending_score']:.3f}",
                'Total Posts': metrics['total_posts'],
                'Total Views': f"{metrics['total_views']:,}"
            })
        
        scorecard_df = pd.DataFrame(scorecard_data)
        
        # Sort by lending score
        scorecard_df = scorecard_df.sort_values('Lending Score', ascending=False)
        
        # Create HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>EthioMart Vendor Scorecard for Micro-Lending</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2E86AB; text-align: center; }}
                h2 {{ color: #A23B72; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .high-score {{ background-color: #d4edda; }}
                .medium-score {{ background-color: #fff3cd; }}
                .low-score {{ background-color: #f8d7da; }}
                .metric {{ margin: 10px 0; }}
                .top-vendor {{ background-color: #e7f3ff; }}
            </style>
        </head>
        <body>
            <h1>EthioMart Vendor Scorecard for Micro-Lending</h1>
            <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <h2>Executive Summary</h2>
            <div class="metric">
                <strong>Total Vendors Analyzed:</strong> {len(scorecard_df)}
            </div>
            <div class="metric">
                <strong>Top Vendor:</strong> {scorecard_df.iloc[0]['Vendor Channel']} 
                (Lending Score: {scorecard_df.iloc[0]['Lending Score']})
            </div>
            <div class="metric">
                <strong>Average Lending Score:</strong> {scorecard_df['Lending Score'].astype(float).mean():.3f}
            </div>
            
            <h2>Vendor Scorecard</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Vendor Channel</th>
                    <th>Avg. Views/Post</th>
                    <th>Posts/Week</th>
                    <th>Avg. Price (ETB)</th>
                    <th>Product Diversity</th>
                    <th>Consistency</th>
                    <th>Lending Score</th>
                    <th>Total Posts</th>
                </tr>
        """
        
        for idx, row in scorecard_df.iterrows():
            lending_score = float(row['Lending Score'])
            
            # Determine score class
            if lending_score >= 0.7:
                score_class = "high-score"
            elif lending_score >= 0.4:
                score_class = "medium-score"
            else:
                score_class = "low-score"
            
            # Add top vendor class
            row_class = "top-vendor" if idx == 0 else score_class
            
            html_report += f"""
                <tr class="{row_class}">
                    <td>{scorecard_df.index.get_loc(idx) + 1}</td>
                    <td>{row['Vendor Channel']}</td>
                    <td>{row['Avg. Views/Post']}</td>
                    <td>{row['Posts/Week']}</td>
                    <td>{row['Avg. Price (ETB)']}</td>
                    <td>{row['Product Diversity']}</td>
                    <td>{row['Consistency Score']}</td>
                    <td>{row['Lending Score']}</td>
                    <td>{row['Total Posts']}</td>
                </tr>
            """
        
        html_report += """
            </table>
            
            <h2>Detailed Vendor Analysis</h2>
        """
        
        # Add detailed analysis for top 3 vendors
        for i in range(min(3, len(scorecard_df))):
            vendor_name = scorecard_df.iloc[i]['Vendor Channel']
            metrics = self.vendor_metrics[vendor_name]
            
            html_report += f"""
            <h3>{i+1}. {vendor_name}</h3>
            <div class="metric">
                <strong>Lending Score:</strong> {metrics['lending_score']:.3f}
            </div>
            <div class="metric">
                <strong>Business Activity:</strong> {metrics['posting_frequency']:.1f} posts/week 
                with {metrics['avg_views_per_post']:.0f} average views per post
            </div>
            <div class="metric">
                <strong>Price Range:</strong> Average {metrics['avg_price_point']:.0f} ETB
            </div>
            """
            
            if metrics['top_performing_post']:
                top_post = metrics['top_performing_post']
                html_report += f"""
                <div class="metric">
                    <strong>Top Performing Post:</strong> {top_post['views']} views<br>
                    <em>"{top_post['message']}"</em>
                </div>
                """
            
            html_report += "<hr>"
        
        html_report += """
            <h2>Scoring Methodology</h2>
            <p>The lending score is calculated using the following weighted metrics:</p>
            <ul>
                <li><strong>Average Views per Post (30%):</strong> Indicates market reach and customer interest</li>
                <li><strong>Posting Frequency (20%):</strong> Shows business activity and consistency</li>
                <li><strong>Average Price Point (20%):</strong> Reflects business scale and target market</li>
                <li><strong>Product Diversity (10%):</strong> Indicates business adaptability</li>
                <li><strong>Engagement Trend (10%):</strong> Shows growth trajectory</li>
                <li><strong>Consistency Score (10%):</strong> Reflects operational reliability</li>
            </ul>
            
            <h2>Recommendations</h2>
            <h3>High-Priority Lending Candidates (Score ≥ 0.7)</h3>
            <p>These vendors show strong business activity, consistent engagement, and good market reach. 
            They are recommended for immediate micro-lending consideration.</p>
            
            <h3>Medium-Priority Candidates (Score 0.4-0.7)</h3>
            <p>These vendors show potential but may need additional evaluation or capacity building 
            before loan approval.</p>
            
            <h3>Low-Priority Candidates (Score < 0.4)</h3>
            <p>These vendors may benefit from business development support before being considered 
            for lending programs.</p>
            
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"Vendor scorecard saved to {output_path}")
        
        # Also save CSV for further analysis
        csv_path = output_path.replace('.html', '.csv')
        scorecard_df.to_csv(csv_path, index=False)
        print(f"Scorecard data saved to {csv_path}")
        
        return scorecard_df
    
    def plot_vendor_analysis(self):
        """
        Create visualizations for vendor analysis
        """
        if not self.vendor_metrics:
            self.calculate_vendor_metrics()
            self.calculate_lending_scores()
        
        # Prepare data for plotting
        vendors = list(self.vendor_metrics.keys())
        lending_scores = [self.vendor_metrics[v]['lending_score'] for v in vendors]
        avg_views = [self.vendor_metrics[v]['avg_views_per_post'] for v in vendors]
        posting_freq = [self.vendor_metrics[v]['posting_frequency'] for v in vendors]
        avg_prices = [self.vendor_metrics[v]['avg_price_point'] for v in vendors]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Lending Scores
        axes[0, 0].bar(range(len(vendors)), lending_scores, color='skyblue')
        axes[0, 0].set_title('Vendor Lending Scores')
        axes[0, 0].set_xlabel('Vendors')
        axes[0, 0].set_ylabel('Lending Score')
        axes[0, 0].set_xticks(range(len(vendors)))
        axes[0, 0].set_xticklabels([v[:15] + '...' if len(v) > 15 else v for v in vendors], rotation=45)
        
        # Plot 2: Views vs Posting Frequency
        axes[0, 1].scatter(posting_freq, avg_views, c=lending_scores, cmap='viridis', s=100)
        axes[0, 1].set_title('Engagement vs Activity')
        axes[0, 1].set_xlabel('Posts per Week')
        axes[0, 1].set_ylabel('Average Views per Post')
        cbar1 = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
        cbar1.set_label('Lending Score')
        
        # Plot 3: Price Distribution
        axes[1, 0].hist([p for p in avg_prices if p > 0], bins=10, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Average Price Distribution')
        axes[1, 0].set_xlabel('Average Price (ETB)')
        axes[1, 0].set_ylabel('Number of Vendors')
        
        # Plot 4: Lending Score vs Average Price
        valid_prices = [(i, p) for i, p in enumerate(avg_prices) if p > 0]
        if valid_prices:
            valid_indices, valid_price_values = zip(*valid_prices)
            valid_scores = [lending_scores[i] for i in valid_indices]
            
            axes[1, 1].scatter(valid_price_values, valid_scores, color='green', alpha=0.6, s=100)
            axes[1, 1].set_title('Lending Score vs Average Price')
            axes[1, 1].set_xlabel('Average Price (ETB)')
            axes[1, 1].set_ylabel('Lending Score')
        
        plt.tight_layout()
        plt.savefig('vendor_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage example
def run_vendor_analysis():
    """
    Main function to run vendor analysis
    """
    # Initialize the analytics engine
    # Update the path to your actual data file
    engine = VendorAnalyticsEngine(
        telegram_data_path="path/to/your/telegram_data.csv",  # Update this path
        ner_model=None  # Add your NER model instance here if available
    )
    
    # Extract entities and calculate metrics
    print("Extracting entities from messages...")
    entities_df = engine.extract_entities_from_messages()
    
    print("Calculating vendor metrics...")
    vendor_metrics = engine.calculate_vendor_metrics()
    
    print("Calculating lending scores...")
    lending_scores = engine.calculate_lending_scores()
    
    # Create vendor scorecard
    print("Creating vendor scorecard...")
    scorecard = engine.create_vendor_scorecard("vendor_scorecard.html")
    
    # Create visualizations
    print("Creating visualizations...")
    engine.plot_vendor_analysis()
    
    # Print summary
    print("\n" + "="*50)
    print("VENDOR ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"Total vendors analyzed: {len(vendor_metrics)}")
    print(f"Average lending score: {np.mean(list(lending_scores.values())):.3f}")
    
    # Top 3 vendors
    sorted_vendors = sorted(
        lending_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    print("\nTop 3 Lending Candidates:")
    for i, (vendor, score) in enumerate(sorted_vendors[:3]):
        metrics = vendor_metrics[vendor]
        print(f"{i+1}. {vendor}")
        print(f"   Lending Score: {score:.3f}")
        print(f"   Avg Views/Post: {metrics['avg_views_per_post']:.0f}")
        print(f"   Posts/Week: {metrics['posting_frequency']:.1f}")
        print(f"   Avg Price: {metrics['avg_price_point']:.0f} ETB")
        print()
    
    return engine, scorecard

if __name__ == "__main__":
    engine, scorecard = run_vendor_analysis()