# Script for initial entity extraction (e.g., rule-based or pre-trained model)
import json
from typing import List, Dict, Tuple
from pathlib import Path

class InteractiveAnnotator:
    def __init__(self):
        self.current_annotations = {}
        self.annotation_history = []
    
    def annotate_batch_interactive(self, messages: List[Dict], batch_size: int = 10):
        """Interactively annotate a batch of messages"""
        annotations = []
        
        for i, message in enumerate(messages[:batch_size]):
            print(f"\n--- Message {i+1}/{min(batch_size, len(messages))} ---")
            print(f"Channel: {message.get('channel_username', 'Unknown')}")
            print(f"Text: {message['cleaned_text']}")
            print(f"Auto-detected entities: {message.get('entity_hints', {})}")
            
            # Get user input for annotation
            annotation = self._get_user_annotation(message['cleaned_text'])
            annotations.append({
                'message_id': message.get('message_id', i),
                'text': message['cleaned_text'],
                'annotation': annotation
            })
            
            # Ask if user wants to continue
            if i < batch_size - 1:
                continue_choice = input("\nContinue to next message? (y/n): ").lower()
                if continue_choice != 'y':
                    break
        
        return annotations
    
    def _get_user_annotation(self, text: str) -> List[Tuple[str, str]]:
        """Get manual annotation for a single text"""
        tokens = text.split()
        annotations = []
        
        print("\nTokens with indices:")
        for i, token in enumerate(tokens):
            print(f"{i}: {token}")
        
        print("\nEntity types: PRODUCT, LOCATION, PRICE")
        print("Format: 'start_idx-end_idx ENTITY_TYPE' (e.g., '0-1 PRODUCT')")
        print("Enter 'done' when finished, 'skip' to skip this message")
        
        while True:
            user_input = input("Enter entity span: ").strip()
            
            if user_input.lower() == 'done':
                break
            elif user_input.lower() == 'skip':
                return []
            
            try:
                parts = user_input.split()
                if len(parts) != 2:
                    print("Invalid format. Use: 'start_idx-end_idx ENTITY_TYPE'")
                    continue
                
                span, entity_type = parts
                start_idx, end_idx = map(int, span.split('-'))
                
                if entity_type.upper() not in ['PRODUCT', 'LOCATION', 'PRICE']:
                    print("Invalid entity type. Use: PRODUCT, LOCATION, or PRICE")
                    continue
                
                if 0 <= start_idx <= end_idx < len(tokens):
                    entity_text = ' '.join(tokens[start_idx:end_idx+1])
                    annotations.append((entity_text, entity_type.upper()))
                    print(f"Added: '{entity_text}' as {entity_type.upper()}")
                else:
                    print("Invalid indices")
                    
            except ValueError:
                print("Invalid format. Use: 'start_idx-end_idx ENTITY_TYPE'")
        
        return annotations