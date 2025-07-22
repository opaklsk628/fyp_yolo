import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:zxcv1234@localhost/project_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
lying_down_frames = 0
LYING_DOWN_THRESHOLD = 30
background_processing = True
last_telegram_notification = 0
TELEGRAM_COOLDOWN = 300
system_running = True
current_resolution = '720p'  # Changed default to 720p
resolution_lock = threading.Lock()
restart_camera_flag = False

# Camera location - can be made configurable later
CAMERA_LOCATION = "G/F Lobby Cam1"

resolution_settings = {
    '240p': (426, 240),    # Ultra low bandwidth
    '360p': (640, 360),    # Low bandwidth
    '480p': (854, 480),    # Medium bandwidth
    '720p': (1280, 720),   # High bandwidth
    '1080p': (1920, 1080)  # Ultra high bandwidth
}

# Create directories
os.makedirs('static/log_photo', exist_ok=True)

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
    # camera_location removed - will handle in display

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
    
    # Release camera
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    
    # Wait for threads to finish
    if camera_thread and camera_thread.is_alive():
        camera_thread.join(timeout=2)
    
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Telegram notification function
def send_telegram_notification(message, image_path=None):
    """Send notification to Telegram"""
    try:
        settings = SystemSettings.query.first()
        if not settings or not settings.telegram_bot_token or not settings.telegram_chat_id:
            print("Telegram settings not configured")
            return False
        
        if not settings.notification_enabled:
            print("Telegram notifications disabled")
            return False
        
        bot_token = settings.telegram_bot_token
        chat_id = settings.telegram_chat_id
        
        if image_path and os.path.exists(image_path):
            # Send photo with caption
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': chat_id,
                    'caption': message,
                    'parse_mode': 'HTML'
                }
                response = requests.post(url, files=files, data=data)
        else:
            # Send text only
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            print("Telegram notification sent successfully")
            return True
        else:
            print(f"Failed to send Telegram notification: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error sending Telegram notification: {e}")
        return False

# Hong Kong Observatory API functions
def get_hko_weather():
    """Get weather data from Hong Kong Observatory"""
    try:
        # Local Weather Forecast
        forecast_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=flw&lang=en"
        forecast_response = requests.get(forecast_url, timeout=5)
        forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None
        
        # Current Weather Report
        current_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en"
        current_response = requests.get(current_url, timeout=5)
        current_data = current_response.json() if current_response.status_code == 200 else None
        
        # Weather Warning Summary
        warning_url = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warnsum&lang=en"
        warning_response = requests.get(warning_url, timeout=5)
        warning_data = warning_response.json() if warning_response.status_code == 200 else None
        
        weather_info = {
            'current': None,
            'forecast': [],
            'warnings': []
        }
        
        # Process current temperature
        if current_data and 'temperature' in current_data:
            temps = current_data['temperature']['data']
            if temps:
                avg_temp = sum(t['value'] for t in temps) / len(temps)
                weather_info['current'] = {
                    'temperature': round(avg_temp, 1),
                    'humidity': current_data.get('humidity', {}).get('data', [{}])[0].get('value', 'N/A')
                }
        
        # Process 7-day forecast
        if forecast_data and 'weatherForecast' in forecast_data:
            for day in forecast_data['weatherForecast'][:7]:
                weather_info['forecast'].append({
                    'date': day.get('forecastDate', ''),
                    'week': day.get('week', ''),
                    'min_temp': day.get('forecastMintemp', {}).get('value', 'N/A'),
                    'max_temp': day.get('forecastMaxtemp', {}).get('value', 'N/A'),
                    'weather': day.get('forecastWeather', '')
                })
        
        # Process warnings
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

# Pose detection functions
def classify_pose(landmarks):
    """Classify pose as standing/sitting or lying down"""
    if not landmarks:
        return "unknown"
    
    # Get key points
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    # Calculate average positions
    hip_y = (left_hip.y + right_hip.y) / 2
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    
    # Check if person is lying down (horizontal position)
    vertical_diff = abs(shoulder_y - hip_y)
    
    # Also check the overall body angle
    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    hip_x = (left_hip.x + right_hip.x) / 2
    horizontal_diff = abs(shoulder_x - hip_x)
    
    # If horizontal difference is large and vertical difference is small, person is lying
    if horizontal_diff > vertical_diff * 2:
        return "lying_down"
    else:
        return "standing/sitting"

def restart_camera():
    """Safely restart camera with new resolution"""
    global camera, restart_camera_flag
    
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        
        time.sleep(0.5)  # Short wait
        
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            width, height = resolution_settings[current_resolution]
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Set FPS based on resolution
            if current_resolution in ['240p', '360p']:
                camera.set(cv2.CAP_PROP_FPS, 10)
            elif current_resolution == '480p':
                camera.set(cv2.CAP_PROP_FPS, 15)
            else:  # 720p, 1080p
                camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Camera restarted with resolution: {current_resolution}")
            return True
    return False

def process_camera_background():
    """Background thread for continuous camera processing"""
    global camera, pose_counts, fall_detected, camera_connected, lying_down_frames
    global background_processing, last_telegram_notification, latest_frame, latest_processed_frame
    global system_running, restart_camera_flag
    
    print("Starting background camera processing...")
    retry_count = 0
    max_retries = 5
    frame_count = 0
    
    while background_processing and system_running:
        try:
            # Check if camera needs restart
            if restart_camera_flag:
                restart_camera_flag = False
                if restart_camera():
                    camera_connected = True
                else:
                    camera_connected = False
                    continue
            
            # Try to open camera if not already open
            if camera is None:
                with camera_lock:
                    print(f"Attempting to open camera... (Attempt {retry_count + 1}/{max_retries})")
                    camera = cv2.VideoCapture(0)
                    
                    # Set camera properties
                    width, height = resolution_settings[current_resolution]
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    # Set FPS based on resolution
                    if current_resolution in ['240p', '360p']:
                        camera.set(cv2.CAP_PROP_FPS, 10)
                    elif current_resolution == '480p':
                        camera.set(cv2.CAP_PROP_FPS, 15)
                    else:  # 720p, 1080p
                        camera.set(cv2.CAP_PROP_FPS, 30)
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
                        print("Camera opened successfully")
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
            
            # Resize frame to current resolution if needed
            height, width = frame.shape[:2]
            target_width, target_height = resolution_settings[current_resolution]
            if width != target_width or height != target_height:
                frame = cv2.resize(frame, (target_width, target_height))
            
            # Update latest frame for streaming
            with camera_lock:
                latest_frame = frame.copy()
            
            # Process every 5th frame to reduce CPU usage
            frame_count += 1
            if frame_count % 5 != 0:
                time.sleep(0.05)  # Adjust sleep based on FPS
                continue
            
            # Process frame with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            # Convert back to BGR for display
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Reset counts for this frame
            pose_counts = {'standing/sitting': 0, 'lying_down': 0}
            
            if results.pose_landmarks:
                # Draw pose
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                
                # Classify pose
                pose_type = classify_pose(results.pose_landmarks.landmark)
                pose_counts[pose_type] = pose_counts.get(pose_type, 0) + 1
                
                # Check for fall detection
                if pose_type == "lying_down":
                    lying_down_frames += 1
                    if lying_down_frames >= LYING_DOWN_THRESHOLD and not fall_detected:
                        fall_detected = True
                        current_time = time.time()
                        
                        # Save fall image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"fall_{timestamp}.jpg"
                        filepath = os.path.join('static', 'log_photo', filename)
                        cv2.imwrite(filepath, frame)
                        
                        # Save to database
                        try:
                            fall_record = FallRecord(
                                lying_down_count=pose_counts['lying_down'],
                                image_path=f"log_photo/{filename}"
                            )
                            db.session.add(fall_record)
                            db.session.commit()
                            
                            # Send Telegram notification with cooldown
                            if current_time - last_telegram_notification > TELEGRAM_COOLDOWN:
                                message = (
                                    "⚠️ <b>Fall Detected!</b>\n"
                                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    f"Location: {CAMERA_LOCATION}\n"
                                    f"Number of people lying down: {pose_counts['lying_down']}\n"
                                    "Please check immediately!"
                                )
                                send_telegram_notification(message, filepath)
                                last_telegram_notification = current_time
                                
                        except Exception as e:
                            print(f"Error saving fall record: {e}")
                            db.session.rollback()
                else:
                    lying_down_frames = 0
                    fall_detected = False
            
            # Add overlays
            cv2.putText(image, f"Location: {CAMERA_LOCATION}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Standing/Sitting: {pose_counts.get('standing/sitting', 0)}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Lying Down: {pose_counts.get('lying_down', 0)}", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if fall_detected:
                cv2.putText(image, "FALL DETECTED!", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Update processed frame
            with camera_lock:
                latest_processed_frame = image.copy()
            
            # Garbage collection every 100 frames
            if frame_count % 100 == 0:
                gc.collect()
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Error in background processing: {e}")
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
    """Generate frames for live video feed with ultra-low bandwidth"""
    global camera, pose_counts, fall_detected, camera_connected, latest_processed_frame
    
    # JPEG quality settings based on resolution
    quality_settings = {
        '240p': 30,   # Ultra low quality
        '360p': 40,   # Low quality
        '480p': 50,   # Medium quality
        '720p': 60,   # Higher quality
        '1080p': 70   # High quality
    }
    
    while True:
        if latest_processed_frame is None:
            # Generate a black frame with "No camera" message
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
        
        # Use the latest processed frame
        with camera_lock:
            frame = latest_processed_frame.copy()
        
        # Get appropriate quality for current resolution
        quality = quality_settings.get(current_resolution, 50)
        
        # Encode with dynamic quality based on resolution
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Dynamic delay based on resolution
        if current_resolution in ['240p', '360p']:
            time.sleep(0.1)  # 10 FPS for low res
        elif current_resolution == '480p':
            time.sleep(0.067)  # 15 FPS for medium res
        else:
            time.sleep(0.033)  # 30 FPS for high res

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
        
        # First user becomes admin
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
    return render_template('dashboard.html', weather=weather)

@app.route('/monitor')
@login_required
def monitor():
    weather = get_hko_weather()
    # Get recent 5 fall records
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
    global pose_counts, fall_detected, camera_connected
    
    return jsonify({
        'standing_sitting': pose_counts.get('standing/sitting', 0),
        'lying_down': pose_counts.get('lying_down', 0),
        'fall_detected': fall_detected,
        'camera_connected': camera_connected,
        'current_resolution': current_resolution,
        'camera_location': CAMERA_LOCATION
    })

@app.route('/set_resolution', methods=['POST'])
@login_required
def set_resolution():
    global current_resolution, restart_camera_flag
    
    data = request.get_json()
    new_resolution = data.get('resolution', '720p')
    
    if new_resolution in resolution_settings:
        with resolution_lock:
            current_resolution = new_resolution
            restart_camera_flag = True  # Flag to restart camera
        
        return jsonify({'success': True, 'resolution': current_resolution})
    else:
        return jsonify({'success': False, 'message': 'Invalid resolution'}), 400

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

# Weather API endpoint
@app.route('/api/weather')
def api_weather():
    weather = get_hko_weather()
    if weather:
        return jsonify({'success': True, 'data': weather})
    else:
        return jsonify({'success': False, 'message': 'Failed to fetch weather data'}), 500

# API endpoint for recent falls (limit to 5)
@app.route('/api/recent_falls')
def api_recent_falls():
    """Get recent fall records for live updates"""
    try:
        records = FallRecord.query.order_by(FallRecord.detection_time.desc()).limit(5).all()
        return jsonify({
            'success': True,
            'falls': [{
                'id': record.id,
                'time': record.detection_time.strftime('%H:%M:%S'),
                'date': record.detection_time.strftime('%Y-%m-%d'),
                'lying_down_count': record.lying_down_count,
                'camera_location': CAMERA_LOCATION  # Use static location
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
        
        # Delete image file
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
    
    # Test notification
    if enabled and bot_token and chat_id:
        test_message = "✅ Telegram notification test successful!\nHeatstroke Detection System is now connected."
        if send_telegram_notification(test_message):
            flash('Telegram settings saved and test message sent!')
        else:
            flash('Telegram settings saved but test message failed. Please check your settings.')
    else:
        flash('Telegram settings saved!')
    
    return redirect(url_for('settings'))

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
        global background_processing, camera, camera_thread, lying_down_frames, fall_detected
        
        print("System restart requested...")
        
        # Stop current processing
        background_processing = False
        
        # Release camera
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None
        
        # Wait for thread to finish
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=5)
        
        # Reset states
        lying_down_frames = 0
        fall_detected = False
        
        # Wait a moment
        time.sleep(2)
        
        # Restart processing
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
        
        # Stop processing
        background_processing = False
        system_running = False
        
        # The signal handler will handle cleanup
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
            # Create session token
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
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 409
        
        # Check if this is the first user (make them admin)
        is_first_user = User.query.count() == 0
        
        # Create new user
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
                'detection_time': record.detection_time.strftime('%Y-%m-%d %H:%M:%S'),
                'lying_down_count': record.lying_down_count,
                'image_path': record.image_path,
                'camera_location': CAMERA_LOCATION  # Use static location
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
            
            # Test notification
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
        user.password_hash = generate_password_hash('123456')
        db.session.commit()
        flash(f'Password reset for {user.username}')
    
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
        # Delete image file
        image_path = os.path.join('static', record.image_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        db.session.delete(record)
        db.session.commit()
        flash('Record deleted')
    
    return redirect(url_for('fall_records'))

if __name__ == '__main__':
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Create default admin if no users exist
        if User.query.count() == 0:
            admin = User(
                username='admin',
                password_hash=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created: admin/admin123")
        
        # Create default system settings if not exist
        if SystemSettings.query.count() == 0:
            settings = SystemSettings()
            db.session.add(settings)
            db.session.commit()
            print("Default system settings created")
        
        # Start background camera processing thread
        camera_thread = threading.Thread(target=process_camera_background, daemon=True)
        camera_thread.start()
        print("Background camera processing started")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


