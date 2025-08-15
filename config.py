import os
from datetime import timedelta

class Config:
    """Application configuration"""
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://postgres:zxcv1234@localhost/project_database'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Upload
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Camera settings
    CAMERA_LOCATION = "G/F Lobby Cam1"
    LYING_DOWN_SECONDS = 5
    FALL_RECORD_COOLDOWN = 30
    MAX_TRACKING_AGE = 2
    
    # Resolution settings
    RESOLUTION_SETTINGS = {
        '240p': (426, 240),
        '360p': (640, 360),
        '480p': (854, 480),
        '720p': (1280, 720),
        '1080p': (1920, 1080)
    }
    
    FPS_SETTINGS = {
        '240p': 10,
        '360p': 10,
        '480p': 15,
        '720p': 20,
        '1080p': 20
    }
    
    DEFAULT_RESOLUTION = '720p'
