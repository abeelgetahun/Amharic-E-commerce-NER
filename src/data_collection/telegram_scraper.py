import asyncio
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from telethon import TelegramClient
from telethon.errors import FloodWaitError, ChannelPrivateError
import yaml
from pathlib import Path

class TelegramScraper:
    def __init__(self, api_id: str, api_hash: str, config_path: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.client = TelegramClient('telegram_scraper', api_id, api_hash)
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('telegram_scraper')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('logs/scraping.log', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    async def start_session(self):
        """Initialize Telegram client session"""
        await self.client.start()
        self.logger.info("Telegram client session started successfully")
    
    async def scrape_channel(self, channel_username: str, max_messages: int = 1000) -> List[Dict]:
        """Scrape messages from a specific channel"""
        messages_data = []
        
        try:
            entity = await self.client.get_entity(channel_username)
            self.logger.info(f"Starting to scrape channel: {channel_username}")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config['scraping_config']['date_range_days'])
            
            message_count = 0
            async for message in self.client.iter_messages(entity, limit=max_messages):
                if message.date < start_date:
                    break
                    
                # Extract message data
                message_data = await self._extract_message_data(message, channel_username)
                if message_data:
                    messages_data.append(message_data)
                    message_count += 1
                
                # Rate limiting
                await asyncio.sleep(self.config['scraping_config']['delay_between_requests'])
                
                if message_count % 100 == 0:
                    self.logger.info(f"Scraped {message_count} messages from {channel_username}")
            
            self.logger.info(f"Completed scraping {channel_username}: {len(messages_data)} messages")
            return messages_data
            
        except ChannelPrivateError:
            self.logger.error(f"Channel {channel_username} is private or doesn't exist")
            return []
        except FloodWaitError as e:
            self.logger.warning(f"Rate limit hit, waiting {e.seconds} seconds")
            await asyncio.sleep(e.seconds)
            return await self.scrape_channel(channel_username, max_messages)
        except Exception as e:
            self.logger.error(f"Error scraping {channel_username}: {str(e)}")
            return []
    
    async def _extract_message_data(self, message, channel_username: str) -> Optional[Dict]:
        """Extract relevant data from a message"""
        if not message.text and not message.media:
            return None
        
        message_data = {
            'message_id': message.id,
            'channel_username': channel_username,
            'text': message.text or '',
            'date': message.date.isoformat(),
            'views': getattr(message, 'views', 0),
            'forwards': getattr(message, 'forwards', 0),
            'replies': getattr(message.replies, 'replies', 0) if message.replies else 0,
            'has_media': bool(message.media),
            'media_type': str(type(message.media).__name__) if message.media else None,
            'sender_id': getattr(message.sender, 'id', None) if message.sender else None,
            'is_bot': getattr(message.sender, 'bot', False) if message.sender else False
        }
        
        # Handle media if present
        if message.media and self.config['scraping_config']['include_media']:
            media_info = await self._extract_media_info(message)
            message_data.update(media_info)
        
        return message_data
    
    async def _extract_media_info(self, message) -> Dict:
        """Extract media information from message"""
        media_info = {}
        
        if hasattr(message.media, 'photo'):
            media_info['media_type'] = 'photo'
            media_info['has_image'] = True
        elif hasattr(message.media, 'document'):
            doc = message.media.document
            media_info['media_type'] = 'document'
            media_info['file_size'] = doc.size
            media_info['mime_type'] = doc.mime_type
        
        return media_info
    
    async def scrape_all_channels(self) -> Dict[str, List[Dict]]:
        """Scrape all configured channels"""
        all_data = {}
        
        for channel_config in self.config['telegram_channels']:
            username = channel_config['username']
            max_messages = self.config['scraping_config']['max_messages_per_channel']
            
            self.logger.info(f"Starting scrape for channel: {username}")
            channel_data = await self.scrape_channel(username, max_messages)
            all_data[username] = channel_data
            
            # Save individual channel data
            self._save_channel_data(username, channel_data)
        
        return all_data
    
    def _save_channel_data(self, channel_username: str, data: List[Dict]):
        """Save scraped data for individual channel"""
        output_dir = Path(f"data/raw/telegram_scrapes/{channel_username}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output_dir / f"{channel_username}_messages.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Save as CSV for easy analysis
        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_dir / f"{channel_username}_messages.csv", index=False, encoding='utf-8')
        
        self.logger.info(f"Saved {len(data)} messages for channel {channel_username}")
    
    async def close_session(self):
        """Close Telegram client session"""
        await self.client.disconnect()
        self.logger.info("Telegram client session closed")