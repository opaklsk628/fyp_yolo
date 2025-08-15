import os
import requests
import traceback
from models.database import SystemSettings

class NotificationService:
    def send_telegram_notification(self, message, image_path=None):
        """Send notification to Telegram"""
        try:
            print("DEBUG: Starting Telegram notification...")
            
            # Import inside method to avoid circular import
            from app import create_app
            app = create_app()
            
            with app.app_context():
                settings = SystemSettings.query.first()
                if not settings:
                    print("DEBUG: No system settings found")
                    return False
                    
                if not settings.telegram_bot_token or not settings.telegram_chat_id:
                    print(f"DEBUG: Missing Telegram settings - Token: {'Yes' if settings.telegram_bot_token else 'No'}, Chat ID: {'Yes' if settings.telegram_chat_id else 'No'}")
                    return False
                
                if not settings.notification_enabled:
                    print("DEBUG: Notifications disabled")
                    return False
                
                bot_token = settings.telegram_bot_token
                chat_id = settings.telegram_chat_id
                
                print(f"DEBUG: Using Bot Token: {bot_token[:10]}... and Chat ID: {chat_id}")
                
                if image_path and os.path.exists(image_path):
                    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                    print(f"DEBUG: Sending photo to URL: {url}")
                    
                    with open(image_path, 'rb') as photo:
                        files = {'photo': photo}
                        data = {
                            'chat_id': chat_id,
                            'caption': message,
                            'parse_mode': 'HTML'
                        }
                        response = requests.post(url, files=files, data=data, timeout=30)
                else:
                    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    print(f"DEBUG: Sending text message to URL: {url}")
                    
                    data = {
                        'chat_id': chat_id,
                        'text': message,
                        'parse_mode': 'HTML'
                    }
                    response = requests.post(url, json=data, timeout=30)
                
                print(f"DEBUG: Telegram API response status: {response.status_code}")
                print(f"DEBUG: Telegram API response: {response.text}")
                
                if response.status_code == 200:
                    print("DEBUG: Telegram notification sent successfully")
                    return True
                else:
                    print(f"DEBUG: Failed to send notification: {response.status_code}, Response: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"ERROR sending Telegram notification: {e}")
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return False
