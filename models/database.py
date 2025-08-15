from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import pytz

db = SQLAlchemy()

HK_TZ = pytz.timezone('Asia/Hong_Kong')
UTC = pytz.UTC

def utc_to_hk(utc_dt):
    """Convert UTC datetime to Hong Kong timezone"""
    if utc_dt.tzinfo is None:
        utc_dt = UTC.localize(utc_dt)
    return utc_dt.astimezone(HK_TZ)

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    session_token = db.Column(db.String(100), nullable=True)
    
    @property
    def password(self):
        return self.password_hash
    
    @password.setter
    def password(self, value):
        self.password_hash = value

    def __repr__(self):
        return f'<User {self.username}>'

class FallRecord(db.Model):
    __tablename__ = 'fall_records'
    id = db.Column(db.Integer, primary_key=True)
    detection_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    lying_down_count = db.Column(db.Integer, nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    camera_location = db.Column(db.String(100), nullable=True)
    
    @property
    def hk_detection_time(self):
        """Get detection time in Hong Kong timezone"""
        return utc_to_hk(self.detection_time)

    def __repr__(self):
        return f'<FallRecord {self.detection_time}>'

class SystemSettings(db.Model):
    __tablename__ = 'system_settings'
    id = db.Column(db.Integer, primary_key=True)
    telegram_bot_token = db.Column(db.String(200), nullable=True)
    telegram_chat_id = db.Column(db.String(100), nullable=True)
    notification_enabled = db.Column(db.Boolean, default=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
