import cv2
import numpy as np
from flask import Flask, render_template, Response, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from sqlalchemy import func, extract
import os
import uuid
import threading
import time
import requests
import json
import signal
import sys
import queue
import gc
import traceback
import pytz
import csv
from io import StringIO, BytesIO
from flask import make_response, send_file
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:zxcv1234@localhost/project_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

HK_TZ = pytz.timezone('Asia/Hong_Kong')
UTC = pytz.UTC

model = YOLO('yolo11s.pt')

# Global variables
camera = None
camera_thread = None
camera_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=2)
latest_frame = None
latest_processed_frame = None
pose_counts = {'standing/sitting': 0, 'lying_down': 0}
fall_detected = False
camera_connected = False
background_processing = True
last_fall_record_time = 0
FALL_RECORD_COOLDOWN = 30
system_running = True
current_resolution = '720p'
resolution_lock = threading.Lock()
restart_camera_flag = False
fall_alert_active = False

# Multi-person tracking
person_states = {}  # {person_id: {'lying_frames': 0, 'last_seen': time, 'bbox': [x1,y1,x2,y2]}}
LYING_DOWN_SECONDS = 5  # 5 seconds for fall detection
MAX_TRACKING_AGE = 2  # Remove person if not seen for 2 seconds

# Camera location
CAMERA_LOCATION = "G/F Lobby Cam1"

resolution_settings = {
    '240p': (426, 240),
    '360p': (640, 360),
    '480p': (854, 480),
    '720p': (1280, 720),
    '1080p': (1920, 1080)
}

fps_settings = {
    '240p': 10,
    '360p': 10,
    '480p': 15,
    '720p': 20,
    '1080p': 20
}

# Create directories
os.makedirs('static/log_photo', exist_ok=True)

# Helper functions for timezone conversion since always change to GMT zone.....
def get_hk_time():
    """Get current time in Hong Kong timezone"""
    return datetime.now(HK_TZ)

def utc_to_hk(utc_dt):
    """Convert UTC datetime to Hong Kong timezone"""
    if utc_dt.tzinfo is None:
        utc_dt = UTC.localize(utc_dt)
    return utc_dt.astimezone(HK_TZ)

def hk_to_utc(hk_dt):
    """Convert Hong Kong datetime to UTC8 for storage"""
    if hk_dt.tzinfo is None:
        hk_dt = HK_TZ.localize(hk_dt)
    return hk_dt.astimezone(UTC).replace(tzinfo=None)

# Database Models
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

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global background_processing, system_running
    print("\nShutting down gracefully...")
    background_processing = False
    system_running = False
    
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    
    if camera_thread and camera_thread.is_alive():
        camera_thread.join(timeout=2)
    
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Telegram notification function
def send_telegram_notification(message, image_path=None):
    """Send notification to Telegram"""
    try:
        print("DEBUG: Starting Telegram notification...")
        
        settings = SystemSettings.query.first()
        if not settings:
            print("DEBUG: No system settings found in database")
            return False
            
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            print(f"DEBUG: Missing Telegram settings - Token: {'Yes' if settings.telegram_bot_token else 'No'}, Chat ID: {'Yes' if settings.telegram_chat_id else 'No'}")
            return False
        
        if not settings.notification_enabled:
            print("DEBUG: Telegram notifications are disabled in settings")
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
            print(f"DEBUG: Failed to send Telegram notification. Status: {response.status_code}, Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR sending Telegram notification: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return False

# Hong Kong Observatory API functions
def get_hko_weather():
    """Get weather data from Hong Kong Observatory"""
    try:
        forecast_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=flw&lang=en"
        forecast_response = requests.get(forecast_url, timeout=5)
        forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None
        
        current_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en"
        current_response = requests.get(current_url, timeout=5)
        current_data = current_response.json() if current_response.status_code == 200 else None
        
        warning_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warnsum&lang=en"
        warning_response = requests.get(warning_url, timeout=5)
        warning_data = warning_response.json() if warning_response.status_code == 200 else None
        
        weather_info = {
            'current': None,
            'forecast': [],
            'warnings': []
        }
        
        if current_data and 'temperature' in current_data:
            temps = current_data['temperature']['data']
            if temps:
                avg_temp = sum(t['value'] for t in temps) / len(temps)
                weather_info['current'] = {
                    'temperature': round(avg_temp, 1),
                    'humidity': current_data.get('humidity', {}).get('data', [{}])[0].get('value', 'N/A')
                }
        
        if forecast_data and 'weatherForecast' in forecast_data:
            for day in forecast_data['weatherForecast'][:7]:
                weather_info['forecast'].append({
                    'date': day.get('forecastDate', ''),
                    'week': day.get('week', ''),
                    'min_temp': day.get('forecastMintemp', {}).get('value', 'N/A'),
                    'max_temp': day.get('forecastMaxtemp', {}).get('value', 'N/A'),
                    'weather': day.get('forecastWeather', '')
                })
        
        if warning_data:
            for warning_type, warning_info in warning_data.items():
                if warning_info and isinstance(warning_info, dict) and warning_info.get('name'):
                    weather_info['warnings'].append({
                        'type': warning_type,
                        'name': warning_info.get('name'),
                        'code': warning_info.get('code', '')
                    })
        
        return weather_info
        
    except Exception as e:
        print(f"Error fetching HKO weather: {e}")
        return None

# Person tracking functions
def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def match_person(bbox, threshold=0.5):
    """Match detected person with existing tracked persons"""
    best_match = None
    best_iou = threshold
    
    for person_id, state in person_states.items():
        iou = calculate_iou(bbox, state['bbox'])
        if iou > best_iou:
            best_iou = iou
            best_match = person_id
    
    return best_match

def is_person_lying(bbox, keypoints=None):
    """Determine if person is lying down based on bbox aspect ratio and position"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Simple heuristic: if width > height * 1.5, person might be lying down
    aspect_ratio = width / height if height > 0 else 0
    
    # Additional check: if the bbox is in lower part of frame (closer to ground)
    # and has high aspect ratio, more likely to be lying down
    is_lying = aspect_ratio > 1.5
    
    return is_lying

def restart_camera():
    """Safely restart camera with new resolution"""
    global camera, restart_camera_flag
    
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        
        time.sleep(0.5)
        
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            width, height = resolution_settings[current_resolution]
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            fps = fps_settings[current_resolution]
            camera.set(cv2.CAP_PROP_FPS, fps)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print(f"Camera restarted with resolution: {current_resolution}, FPS: {fps}")
            return True
    return False

def process_camera_background():
    """Background thread for continuous camera processing with YOLO11"""
    global camera, pose_counts, fall_detected, camera_connected
    global background_processing, latest_frame, latest_processed_frame
    global system_running, restart_camera_flag, last_fall_record_time, fall_alert_active
    global person_states
    
    print("Starting background camera processing with YOLO11...")
    retry_count = 0
    max_retries = 5
    frame_count = 0
    
    while background_processing and system_running:
        try:
            if restart_camera_flag:
                restart_camera_flag = False
                if restart_camera():
                    camera_connected = True
                else:
                    camera_connected = False
                    continue
            
            if camera is None:
                with camera_lock:
                    print(f"Attempting to open camera... (Attempt {retry_count + 1}/{max_retries})")
                    camera = cv2.VideoCapture(0)
                    
                    width, height = resolution_settings[current_resolution]
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    
                    fps = fps_settings[current_resolution]
                    camera.set(cv2.CAP_PROP_FPS, fps)
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if not camera.isOpened():
                        print("Failed to open camera")
                        camera_connected = False
                        camera = None
                        retry_count += 1
                        
                        if retry_count >= max_retries:
                            print(f"Failed to open camera after {max_retries} attempts. Waiting 30 seconds...")
                            time.sleep(30)
                            retry_count = 0
                        else:
                            time.sleep(5)
                        continue
                    else:
                        print(f"Camera opened successfully - FPS: {fps}")
                        camera_connected = True
                        retry_count = 0
            
            success = False
            frame = None
            
            with camera_lock:
                if camera is not None:
                    success, frame = camera.read()
            
            if not success or frame is None:
                print("Failed to read frame from camera")
                camera_connected = False
                with camera_lock:
                    if camera is not None:
                        camera.release()
                        camera = None
                time.sleep(1)
                continue
            
            height, width = frame.shape[:2]
            target_width, target_height = resolution_settings[current_resolution]
            if width != target_width or height != target_height:
                frame = cv2.resize(frame, (target_width, target_height))
            
            with camera_lock:
                latest_frame = frame.copy()
            
            frame_count += 1
            
            # Process frame with YOLO11
            results = model(frame, conf=0.5, classes=[0])  # class 0 is 'person'
            
            # Create visualization frame
            vis_frame = frame.copy()
            
            # Reset counts
            pose_counts = {'standing/sitting': 0, 'lying_down': 0}
            
            # Current time for tracking
            current_time = time.time()
            fps = fps_settings[current_resolution]
            
            # Track detected persons
            detected_persons = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [x1, y1, x2, y2]
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Match with existing persons
                    person_id = match_person(bbox)
                    
                    if person_id is None:
                        # New person detected
                        person_id = f"person_{int(current_time * 1000)}"
                        person_states[person_id] = {
                            'lying_frames': 0,
                            'last_seen': current_time,
                            'bbox': bbox
                        }
                    
                    detected_persons.append(person_id)
                    
                    # Update person state
                    person_states[person_id]['last_seen'] = current_time
                    person_states[person_id]['bbox'] = bbox
                    
                    # Check if person is lying down
                    is_lying = is_person_lying(bbox)
                    
                    if is_lying:
                        person_states[person_id]['lying_frames'] += 1
                        pose_counts['lying_down'] += 1
                        color = (0, 0, 255)  # Red for lying down
                        
                        # Calculate seconds lying down
                        seconds_lying = person_states[person_id]['lying_frames'] / fps
                        
                        # Draw person box and status
                        cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(vis_frame, f"Lying: {seconds_lying:.1f}s", 
                                  (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, color, 2)
                        
                        print(f"DEBUG: {person_id} lying down for {seconds_lying:.1f}s")
                        
                        # Check for fall detection (5 seconds)
                        if seconds_lying >= LYING_DOWN_SECONDS:
                            current_fall_time = time.time()
                            
                            if current_fall_time - last_fall_record_time >= FALL_RECORD_COOLDOWN:
                                fall_detected = True
                                fall_alert_active = True
                                last_fall_record_time = current_fall_time
                                
                                print("DEBUG: Fall detected! Saving record and sending notifications...")
                                
                                # Save fall image
                                hk_time = get_hk_time()
                                timestamp = hk_time.strftime("%Y%m%d_%H%M%S")
                                filename = f"fall_{timestamp}.jpg"
                                filepath = os.path.join('static', 'log_photo', filename)
                                
                                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                                
                                save_success = cv2.imwrite(filepath, frame)
                                print(f"DEBUG: Image save {'successful' if save_success else 'failed'}: {filepath}")
                                
                                # Save to database
                                try:
                                    with app.app_context():
                                        fall_record = FallRecord(
                                            lying_down_count=pose_counts.get('lying_down', 1),
                                            image_path=f"log_photo/{filename}"
                                        )
                                        db.session.add(fall_record)
                                        db.session.commit()
                                        print(f"DEBUG: Database record saved successfully. ID: {fall_record.id}")
                                        
                                        # Send Telegram notification
                                        hk_time_str = hk_time.strftime('%Y-%m-%d %H:%M:%S')
                                        message = (
                                            "🚨 <b>FALL DETECTED!</b> 🚨\n"
                                            f"⏰ Time: {hk_time_str}\n"
                                            f"📍 Location: {CAMERA_LOCATION}\n"
                                            f"👥 Persons lying down: {pose_counts.get('lying_down', 1)}\n"
                                            "⚠️ <b>IMMEDIATE ATTENTION REQUIRED!</b>"
                                        )
                                        notification_sent = send_telegram_notification(message, filepath)
                                        if notification_sent:
                                            print("DEBUG: Telegram notification sent successfully")
                                        else:
                                            print("DEBUG: Telegram notification failed")
                                            
                                except Exception as e:
                                    print(f"ERROR saving fall record: {e}")
                                    print(f"DEBUG: Traceback: {traceback.format_exc()}")
                                    db.session.rollback()
                    else:
                        person_states[person_id]['lying_frames'] = 0
                        pose_counts['standing/sitting'] += 1
                        color = (0, 255, 0)  # Green for standing/sitting
                        
                        cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(vis_frame, "Standing/Sitting", 
                                  (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, color, 2)
            
            # Clean up old tracked persons
            persons_to_remove = []
            for person_id, state in person_states.items():
                if person_id not in detected_persons:
                    if current_time - state['last_seen'] > MAX_TRACKING_AGE:
                        persons_to_remove.append(person_id)
            
            for person_id in persons_to_remove:
                del person_states[person_id]
            
            # Clear fall alert if no one is lying down
            if pose_counts['lying_down'] == 0 and fall_alert_active:
                fall_detected = False
                fall_alert_active = False
                print("DEBUG: Fall alert cleared - no one lying down")
            
            # Add overlays
            cv2.putText(vis_frame, f"Location: {CAMERA_LOCATION}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Total Persons: {len(detected_persons)}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Standing/Sitting: {pose_counts.get('standing/sitting', 0)}", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Lying Down: {pose_counts.get('lying_down', 0)}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if fall_detected:
                cv2.putText(vis_frame, "FALL DETECTED!", (10, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Update processed frame
            with camera_lock:
                latest_processed_frame = vis_frame.copy()
            
            # Garbage collection
            if frame_count % 100 == 0:
                gc.collect()
            
            # Sleep based on FPS
            time.sleep(1.0 / fps)
            
        except Exception as e:
            print(f"Error in background processing: {e}")
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            with camera_lock:
                if camera is not None:
                    camera.release()
                    camera = None
            camera_connected = False
            time.sleep(5)
    
    print("Background camera processing stopped")
    with camera_lock:
        if camera:
            camera.release()
            camera = None

def generate_frames():
    """Generate frames for live video feed"""
    global camera, pose_counts, fall_detected, camera_connected, latest_processed_frame
    
    quality_settings = {
        '240p': 30,
        '360p': 40,
        '480p': 50,
        '720p': 60,
        '1080p': 70
    }
    
    while True:
        if latest_processed_frame is None:
            width, height = resolution_settings[current_resolution]
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera not connected", (width//4, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            quality = quality_settings.get(current_resolution, 50)
            ret, buffer = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
            continue
        
        with camera_lock:
            frame = latest_processed_frame.copy()
        
        quality = quality_settings.get(current_resolution, 50)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        fps = fps_settings[current_resolution]
        time.sleep(1.0 / fps)

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/current_frame')
def current_frame():
    """Provide current frame as a single image"""
    global latest_processed_frame, camera_connected
    
    quality_settings = {
        '240p': 30,
        '360p': 40,
        '480p': 50,
        '720p': 60,
        '1080p': 70
    }
    
    if latest_processed_frame is None:
        width, height = resolution_settings[current_resolution]
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(blank, "Camera not connected", (width//4, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        quality = quality_settings.get(current_resolution, 50)
        ret, buffer = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        with camera_lock:
            frame = latest_processed_frame.copy()
        quality = quality_settings.get(current_resolution, 50)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    response = Response(buffer.tobytes(), mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        is_first_user = User.query.count() == 0
        
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            is_admin=is_first_user
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    weather = get_hko_weather()
    
    today_hk = get_hk_time().date()
    
    today_start_utc = hk_to_utc(HK_TZ.localize(datetime.combine(today_hk, datetime.min.time())))
    today_end_utc = hk_to_utc(HK_TZ.localize(datetime.combine(today_hk, datetime.max.time())))
    
    today_count = db.session.query(func.count(FallRecord.id)).filter(
        FallRecord.detection_time >= today_start_utc,
        FallRecord.detection_time <= today_end_utc
    ).scalar() or 0
    
    month_start = today_hk.replace(day=1)
    month_start_utc = hk_to_utc(HK_TZ.localize(datetime.combine(month_start, datetime.min.time())))
    
    month_count = db.session.query(func.count(FallRecord.id)).filter(
        FallRecord.detection_time >= month_start_utc
    ).scalar() or 0
    
    year_start = today_hk.replace(month=1, day=1)
    year_start_utc = hk_to_utc(HK_TZ.localize(datetime.combine(year_start, datetime.min.time())))
    
    year_count = db.session.query(func.count(FallRecord.id)).filter(
        FallRecord.detection_time >= year_start_utc
    ).scalar() or 0
    
    total_count = db.session.query(func.count(FallRecord.id)).scalar() or 0
    
    return render_template('dashboard.html', 
                         weather=weather,
                         statistics={
                             'today': today_count,
                             'this_month': month_count,
                             'this_year': year_count,
                             'total': total_count
                         })

@app.route('/monitor')
@login_required
def monitor():
    weather = get_hko_weather()
    recent_falls = FallRecord.query.order_by(FallRecord.detection_time.desc()).limit(5).all()
    return render_template('monitor.html', 
                         weather=weather, 
                         current_resolution=current_resolution, 
                         recent_falls=recent_falls,
                         camera_location=CAMERA_LOCATION)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def get_counts():
    global pose_counts, fall_detected, camera_connected, person_states
    
    # Calculate total lying seconds (max among all tracked persons)
    max_lying_seconds = 0
    if person_states:
        fps = fps_settings.get(current_resolution, 30)
        for person_id, state in person_states.items():
            lying_seconds = state['lying_frames'] / fps
            max_lying_seconds = max(max_lying_seconds, lying_seconds)
    
    return jsonify({
        'standing_sitting': pose_counts.get('standing/sitting', 0),
        'lying_down': pose_counts.get('lying_down', 0),
        'fall_detected': fall_detected,
        'camera_connected': camera_connected,
        'current_resolution': current_resolution,
        'camera_location': CAMERA_LOCATION,
        'lying_seconds': round(max_lying_seconds, 1),
        'lying_progress': min(100, int((max_lying_seconds / LYING_DOWN_SECONDS) * 100)),
        'total_persons': len(person_states)
    })

@app.route('/set_resolution', methods=['POST'])
@login_required
def set_resolution():
    global current_resolution, restart_camera_flag
    
    try:
        data = request.get_json()
        new_resolution = data.get('resolution', '720p')
        
        if new_resolution in resolution_settings:
            with resolution_lock:
                current_resolution = new_resolution
                restart_camera_flag = True
            
            print(f"Resolution changed to: {new_resolution}")
            return jsonify({'success': True, 'resolution': current_resolution})
        else:
            return jsonify({'success': False, 'message': 'Invalid resolution'}), 400
    except Exception as e:
        print(f"Error setting resolution: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/fall_records')
@login_required
def fall_records():
    records = FallRecord.query.order_by(FallRecord.detection_time.desc()).all()
    return render_template('fall_records.html', records=records)

@app.route('/settings')
@login_required
def settings():
    if not current_user.is_admin:
        flash('Admin access required')
        return redirect(url_for('dashboard'))
    
    users = User.query.all()
    telegram_settings = SystemSettings.query.first()
    return render_template('settings.html', users=users, telegram_settings=telegram_settings)

# Export routes
@app.route('/export/fall_records')
@login_required
def export_fall_records():
    """Export fall records as CSV"""
    days = request.args.get('days', 30, type=int)
    
    end_date = get_hk_time()
    start_date = end_date - timedelta(days=days)
    start_utc = hk_to_utc(start_date)
    end_utc = hk_to_utc(end_date)
    
    records = FallRecord.query.filter(
        FallRecord.detection_time >= start_utc,
        FallRecord.detection_time <= end_utc
    ).order_by(FallRecord.detection_time.desc()).all()
    
    si = StringIO()
    writer = csv.writer(si)
    
    writer.writerow([
        'Detection Time (HK)',
        'Date',
        'Time',
        'Day of Week',
        'Camera Location',
        'Lying Down Count',
        'Image File'
    ])
    
    for record in records:
        hk_time = record.hk_detection_time
        writer.writerow([
            hk_time.strftime('%Y-%m-%d %H:%M:%S'),
            hk_time.strftime('%Y-%m-%d'),
            hk_time.strftime('%H:%M:%S'),
            hk_time.strftime('%A'),
            CAMERA_LOCATION,
            record.lying_down_count,
            record.image_path
        ])
    
    writer.writerow([])
    writer.writerow(['Summary'])
    writer.writerow(['Total Records:', len(records)])
    writer.writerow(['Date Range:', f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"])
    writer.writerow(['Report Generated:', get_hk_time().strftime('%Y-%m-%d %H:%M:%S HKT')])
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=fall_records_{end_date.strftime('%Y%m%d')}_{days}days.csv"
    output.headers["Content-type"] = "text/csv"
    
    return output

@app.route('/export/statistics')
@login_required
def export_statistics():
    """Export statistics report as CSV"""
    days = request.args.get('days', 30, type=int)
    
    si = StringIO()
    writer = csv.writer(si)
    
    writer.writerow(['Heatstroke Detection System - Statistics Report'])
    writer.writerow(['Generated:', get_hk_time().strftime('%Y-%m-%d %H:%M:%S HKT')])
    writer.writerow(['Period:', f'Last {days} days'])
    writer.writerow([])
    
    writer.writerow(['=== DAILY STATISTICS ==='])
    writer.writerow(['Date', 'Fall Count', 'Day of Week'])
    
    end_date = get_hk_time().date()
    daily_total = 0
    
    for i in range(days):
        date = end_date - timedelta(days=i)
        day_start_utc = hk_to_utc(HK_TZ.localize(datetime.combine(date, datetime.min.time())))
        day_end_utc = hk_to_utc(HK_TZ.localize(datetime.combine(date, datetime.max.time())))
        
        count = db.session.query(func.count(FallRecord.id)).filter(
            FallRecord.detection_time >= day_start_utc,
            FallRecord.detection_time <= day_end_utc
        ).scalar() or 0
        
        daily_total += count
        
        writer.writerow([
            date.strftime('%Y-%m-%d'),
            count,
            date.strftime('%A')
        ])
    
    writer.writerow([])
    writer.writerow(['Daily Average:', round(daily_total / days, 2)])
    writer.writerow([])
    
    writer.writerow(['=== HOURLY DISTRIBUTION ==='])
    writer.writerow(['Hour (HKT)', 'Total Falls', 'Percentage'])
    
    start_utc = hk_to_utc(HK_TZ.localize(datetime.combine(end_date - timedelta(days=days-1), datetime.min.time())))
    end_utc = hk_to_utc(HK_TZ.localize(datetime.combine(end_date, datetime.max.time())))
    
    total_falls = db.session.query(func.count(FallRecord.id)).filter(
        FallRecord.detection_time >= start_utc,
        FallRecord.detection_time <= end_utc
    ).scalar() or 0
    
    for hour in range(24):
        utc_hour = (hour - 8) % 24
        
        count = db.session.query(func.count(FallRecord.id)).filter(
            FallRecord.detection_time >= start_utc,
            FallRecord.detection_time <= end_utc,
            func.extract('hour', FallRecord.detection_time) == utc_hour
        ).scalar() or 0
        
        percentage = (count / total_falls * 100) if total_falls > 0 else 0
        
        writer.writerow([
            f"{hour:02d}:00-{hour:02d}:59",
            count,
            f"{percentage:.1f}%"
        ])
    
    writer.writerow([])
    writer.writerow(['Total Falls in Period:', total_falls])
    
    writer.writerow([])
    writer.writerow(['=== WEEKLY SUMMARY ==='])
    writer.writerow(['Week Starting', 'Fall Count'])
    
    for week in range(0, days, 7):
        if week + 7 > days:
            break
            
        week_start = end_date - timedelta(days=week+6)
        week_end = end_date - timedelta(days=week)
        
        week_start_utc = hk_to_utc(HK_TZ.localize(datetime.combine(week_start, datetime.min.time())))
        week_end_utc = hk_to_utc(HK_TZ.localize(datetime.combine(week_end, datetime.max.time())))
        
        week_count = db.session.query(func.count(FallRecord.id)).filter(
            FallRecord.detection_time >= week_start_utc,
            FallRecord.detection_time <= week_end_utc
        ).scalar() or 0
        
        writer.writerow([
            week_start.strftime('%Y-%m-%d'),
            week_count
        ])
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=statistics_{end_date.strftime('%Y%m%d')}_{days}days.csv"
    output.headers["Content-type"] = "text/csv"
    
    return output

# Weather API endpoint
@app.route('/api/weather')
def api_weather():
    weather = get_hko_weather()
    if weather:
        return jsonify({'success': True, 'data': weather})
    else:
        return jsonify({'success': False, 'message': 'Failed to fetch weather data'}), 500

# Statistics API endpoint
@app.route('/api/statistics')
def api_statistics():
    """Get fall statistics for dashboard"""
    try:
        today_hk = get_hk_time().date()
        
        today_start_utc = hk_to_utc(HK_TZ.localize(datetime.combine(today_hk, datetime.min.time())))
        today_end_utc = hk_to_utc(HK_TZ.localize(datetime.combine(today_hk, datetime.max.time())))
        
        today_count = db.session.query(func.count(FallRecord.id)).filter(
            FallRecord.detection_time >= today_start_utc,
            FallRecord.detection_time <= today_end_utc
        ).scalar() or 0
        
        month_start = today_hk.replace(day=1)
        month_start_utc = hk_to_utc(HK_TZ.localize(datetime.combine(month_start, datetime.min.time())))
        
        month_count = db.session.query(func.count(FallRecord.id)).filter(
            FallRecord.detection_time >= month_start_utc
        ).scalar() or 0
        
        year_start = today_hk.replace(month=1, day=1)
        year_start_utc = hk_to_utc(HK_TZ.localize(datetime.combine(year_start, datetime.min.time())))
        
        year_count = db.session.query(func.count(FallRecord.id)).filter(
            FallRecord.detection_time >= year_start_utc
        ).scalar() or 0
        
        seven_days_ago = today_hk - timedelta(days=6)
        daily_counts = []
        
        for i in range(7):
            date = seven_days_ago + timedelta(days=i)
            day_start_utc = hk_to_utc(HK_TZ.localize(datetime.combine(date, datetime.min.time())))
            day_end_utc = hk_to_utc(HK_TZ.localize(datetime.combine(date, datetime.max.time())))
            
            count = db.session.query(func.count(FallRecord.id)).filter(
                FallRecord.detection_time >= day_start_utc,
                FallRecord.detection_time <= day_end_utc
            ).scalar() or 0
            
            daily_counts.append({
                'date': date.strftime('%Y-%m-%d'),
                'day': date.strftime('%a'),
                'count': count
            })
        
        total_count = db.session.query(func.count(FallRecord.id)).scalar() or 0
        
        thirty_days_ago = today_hk - timedelta(days=29)
        thirty_days_ago_utc = hk_to_utc(HK_TZ.localize(datetime.combine(thirty_days_ago, datetime.min.time())))
        
        thirty_day_count = db.session.query(func.count(FallRecord.id)).filter(
            FallRecord.detection_time >= thirty_days_ago_utc
        ).scalar() or 0
        avg_per_day = round(thirty_day_count / 30, 1)
        
        return jsonify({
            'success': True,
            'statistics': {
                'today': today_count,
                'this_month': month_count,
                'this_year': year_count,
                'total': total_count,
                'average_per_day': avg_per_day,
                'trend_7_days': daily_counts
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# API endpoint for recent falls
@app.route('/api/recent_falls')
def api_recent_falls():
    """Get recent fall records for live updates"""
    try:
        records = FallRecord.query.order_by(FallRecord.detection_time.desc()).limit(5).all()
        return jsonify({
            'success': True,
            'falls': [{
                'id': record.id,
                'time': record.hk_detection_time.strftime('%H:%M:%S'),
                'date': record.hk_detection_time.strftime('%Y-%m-%d'),
                'lying_down_count': record.lying_down_count,
                'camera_location': CAMERA_LOCATION
            } for record in records]
        })
    except Exception as e:
        print(f"Error in api_recent_falls: {e}")
        return jsonify({'success': False, 'message': str(e), 'falls': []}), 200

# Delete fall record (admin only)
@app.route('/api/fall_records/<int:record_id>', methods=['DELETE'])
@login_required
def api_delete_fall_record(record_id):
    """Delete a fall record (admin only)"""
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    
    try:
        record = db.session.get(FallRecord, record_id)
        if not record:
            return jsonify({'success': False, 'message': 'Record not found'}), 404
        
        if record.image_path:
            image_path = os.path.join('static', record.image_path)
            if os.path.exists(image_path):
                os.remove(image_path)
        
        db.session.delete(record)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Record deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

# Telegram settings routes
@app.route('/settings/telegram', methods=['POST'])
@login_required
def update_telegram_settings():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    
    bot_token = request.form.get('telegram_bot_token')
    chat_id = request.form.get('telegram_chat_id')
    enabled = request.form.get('notification_enabled') == 'on'
    
    settings = SystemSettings.query.first()
    if not settings:
        settings = SystemSettings()
        db.session.add(settings)
    
    settings.telegram_bot_token = bot_token
    settings.telegram_chat_id = chat_id
    settings.notification_enabled = enabled
    
    db.session.commit()
    
    if enabled and bot_token and chat_id:
        test_message = "✅ Telegram notification test successful!\nHeatstroke Detection System is now connected."
        if send_telegram_notification(test_message):
            flash('Telegram settings saved and test message sent!')
        else:
            flash('Telegram settings saved but test message failed. Please check your settings.')
    else:
        flash('Telegram settings saved!')
    
    return redirect(url_for('settings'))

# Test fall detection endpoint
@app.route('/test_fall_detection', methods=['POST'])
@login_required
def test_fall_detection():
    """Manually trigger a fall detection for testing"""
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    
    try:
        hk_time = get_hk_time()
        timestamp = hk_time.strftime("%Y%m%d_%H%M%S")
        filename = f"test_fall_{timestamp}.jpg"
        
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "TEST FALL DETECTION", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(test_image, timestamp, (50, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        filepath = os.path.join('static', 'log_photo', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, test_image)
        
        fall_record = FallRecord(
            lying_down_count=1,
            image_path=f"log_photo/{filename}"
        )
        db.session.add(fall_record)
        db.session.commit()
        
        hk_time_str = hk_time.strftime('%Y-%m-%d %H:%M:%S')
        message = (
            "🧪 <b>TEST: Fall Detection</b>\n"
            f"Time: {hk_time_str}\n"
            f"Location: {CAMERA_LOCATION}\n"
            "This is a test notification."
        )
        
        notification_sent = send_telegram_notification(message, filepath)
        
        return jsonify({
            'success': True,
            'message': 'Test fall detection created',
            'record_id': fall_record.id,
            'telegram_sent': notification_sent
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

# System control routes for mobile app
@app.route('/api/system/restart', methods=['POST'])
def api_restart_system():
    """Restart the monitoring system (admin only)"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'success': False, 'message': 'No authorization token'}), 401
    
    token = auth_header.split(' ')[1]
    user = User.query.filter_by(session_token=token).first()
    
    if not user:
        return jsonify({'success': False, 'message': 'Invalid token'}), 401
    
    if not user.is_admin:
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    
    try:
        global background_processing, camera, camera_thread, person_states, fall_detected, fall_alert_active
        
        print("System restart requested...")
        
        background_processing = False
        
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None
        
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=5)
        
        person_states = {}
        fall_detected = False
        fall_alert_active = False
        
        time.sleep(2)
        
        background_processing = True
        camera_thread = threading.Thread(target=process_camera_background, daemon=True)
        camera_thread.start()
        
        print("System restarted successfully")
        return jsonify({'success': True, 'message': 'System restarted'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/system/stop', methods=['POST'])
def api_stop_system():
    """Stop the monitoring system (admin only)"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'success': False, 'message': 'No authorization token'}), 401
    
    token = auth_header.split(' ')[1]
    user = User.query.filter_by(session_token=token).first()
    
    if not user:
        return jsonify({'success': False, 'message': 'Invalid token'}), 401
    
    if not user.is_admin:
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    
    try:
        global background_processing, system_running
        
        print("System stop requested...")
        
        background_processing = False
        system_running = False
        
        os.kill(os.getpid(), signal.SIGTERM)
        
        return jsonify({'success': True, 'message': 'System shutdown initiated'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Resolution settings API for mobile app
@app.route('/api/resolution', methods=['GET', 'POST'])
def api_resolution():
    """Get or set resolution for mobile app"""
    global current_resolution, restart_camera_flag
    
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'success': False, 'message': 'No authorization token'}), 401
    
    token = auth_header.split(' ')[1]
    user = User.query.filter_by(session_token=token).first()
    
    if not user:
        return jsonify({'success': False, 'message': 'Invalid token'}), 401
    
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'current_resolution': current_resolution,
            'available_resolutions': list(resolution_settings.keys())
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        new_resolution = data.get('resolution', '720p')
        
        if new_resolution in resolution_settings:
            with resolution_lock:
                current_resolution = new_resolution
                restart_camera_flag = True
            
            return jsonify({'success': True, 'resolution': current_resolution})
        else:
            return jsonify({'success': False, 'message': 'Invalid resolution'}), 400

# API endpoints for mobile app
@app.route('/api/login', methods=['POST'])
def api_login():
    """Mobile app login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session_token = str(uuid.uuid4())
            user.session_token = session_token
            db.session.commit()
            
            return jsonify({
                'success': True,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'is_admin': user.is_admin,
                    'session_token': session_token
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def api_register():
    """Mobile app registration endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 409
        
        is_first_user = User.query.count() == 0
        
        hashed_password = generate_password_hash(password)
        new_user = User(
            username=username,
            password_hash=hashed_password,
            is_admin=is_first_user
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user': {
                'id': new_user.id,
                'username': new_user.username,
                'is_admin': new_user.is_admin
            }
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/users', methods=['GET'])
def api_get_users():
    """Get all users (admin only)"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'success': False, 'message': 'No authorization token'}), 401
    
    token = auth_header.split(' ')[1]
    user = User.query.filter_by(session_token=token).first()
    
    if not user or not user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        users = User.query.all()
        return jsonify({
            'success': True,
            'users': [{
                'id': user.id,
                'username': user.username,
                'is_admin': user.is_admin
            } for user in users]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/users/<int:user_id>/toggle_admin', methods=['POST'])
def api_toggle_admin(user_id):
    """Toggle admin status (admin only)"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'success': False, 'message': 'No authorization token'}), 401
    
    token = auth_header.split(' ')[1]
    current_user = User.query.filter_by(session_token=token).first()
    
    if not current_user or not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 404
        
        user.is_admin = not user.is_admin
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Admin status updated',
            'is_admin': user.is_admin
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

# API endpoint for fall records
@app.route('/api/fall_records', methods=['GET'])
def api_fall_records():
    """Get fall records in JSON format"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'success': False, 'message': 'No authorization token'}), 401
    
    try:
        records = FallRecord.query.order_by(FallRecord.detection_time.desc()).all()
        return jsonify({
            'success': True,
            'records': [{
                'id': record.id,
                'detection_time': record.hk_detection_time.strftime('%Y-%m-%d %H:%M:%S'),
                'lying_down_count': record.lying_down_count,
                'image_path': record.image_path,
                'camera_location': CAMERA_LOCATION
            } for record in records]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/telegram_settings', methods=['GET', 'POST'])
def api_telegram_settings():
    """Get or update Telegram settings"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'success': False, 'message': 'No authorization token'}), 401
    
    token = auth_header.split(' ')[1]
    user = User.query.filter_by(session_token=token).first()
    
    if not user or not user.is_admin:
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    
    if request.method == 'GET':
        settings = SystemSettings.query.first()
        if settings:
            return jsonify({
                'success': True,
                'settings': {
                    'telegram_bot_token': settings.telegram_bot_token or '',
                    'telegram_chat_id': settings.telegram_chat_id or '',
                    'notification_enabled': settings.notification_enabled
                }
            })
        else:
            return jsonify({
                'success': True,
                'settings': {
                    'telegram_bot_token': '',
                    'telegram_chat_id': '',
                    'notification_enabled': True
                }
            })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            settings = SystemSettings.query.first()
            if not settings:
                settings = SystemSettings()
                db.session.add(settings)
            
            settings.telegram_bot_token = data.get('telegram_bot_token', '')
            settings.telegram_chat_id = data.get('telegram_chat_id', '')
            settings.notification_enabled = data.get('notification_enabled', True)
            
            db.session.commit()
            
            if settings.notification_enabled and settings.telegram_bot_token and settings.telegram_chat_id:
                test_message = "✅ Telegram notification test successful!\nHeatstroke Detection System is now connected."
                send_telegram_notification(test_message)
            
            return jsonify({'success': True, 'message': 'Settings updated successfully'})
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': str(e)}), 500

# Admin functions (web interface)
@app.route('/settings/reset_pw/<int:user_id>', methods=['POST'])
@login_required
def reset_password(user_id):
    if not current_user.is_admin:
        return redirect(url_for('dashboard'))
    
    user = db.session.get(User, user_id)
    if user:
        new_password = '123456'
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        flash(f'Password reset for {user.username}. New password is: {new_password}')
    
    return redirect(url_for('settings'))

@app.route('/settings/toggle_admin/<int:user_id>', methods=['POST'])
@login_required
def toggle_admin(user_id):
    if not current_user.is_admin:
        return redirect(url_for('dashboard'))
    
    user = db.session.get(User, user_id)
    if user and user.id != current_user.id:
        user.is_admin = not user.is_admin
        db.session.commit()
        flash(f'Admin status toggled for {user.username}')
    
    return redirect(url_for('settings'))

@app.route('/settings/delete/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        return redirect(url_for('dashboard'))
    
    user = db.session.get(User, user_id)
    if user and user.id != current_user.id:
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.username} deleted')
    
    return redirect(url_for('settings'))

@app.route('/fall_records/delete/<int:record_id>', methods=['POST'])
@login_required
def delete_fall_record(record_id):
    if not current_user.is_admin:
        return redirect(url_for('fall_records'))
    
    record = db.session.get(FallRecord, record_id)
    if record:
        image_path = os.path.join('static', record.image_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        db.session.delete(record)
        db.session.commit()
        flash('Record deleted')
    
    return redirect(url_for('fall_records'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        if User.query.count() == 0:
            admin = User(
                username='admin',
                password_hash=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created: admin/admin123")
        
        if SystemSettings.query.count() == 0:
            settings = SystemSettings()
            db.session.add(settings)
            db.session.commit()
            print("Default system settings created")
        
        camera_thread = threading.Thread(target=process_camera_background, daemon=True)
        camera_thread.start()
        print("Background camera processing with YOLO11 started")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


