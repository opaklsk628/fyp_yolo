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
pose_counts = {'standing/sitting': 0, 'lying_down': 0}
fall_detected = False
camera_connected = False
lying_down_frames = 0
LYING_DOWN_THRESHOLD = 30  # frames before detecting as fall

# Create directories
os.makedirs('static/log_photo', exist_ok=True)

# Database Models
class User(UserMixin, db.Model):
    __tablename__ = 'users'  # 使用原始的 users 表
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)  # 注意：原始表用 password_hash
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    session_token = db.Column(db.String(100), nullable=True)
    
    # 添加 password 屬性以兼容
    @property
    def password(self):
        return self.password_hash
    
    @password.setter
    def password(self, value):
        self.password_hash = value

    def __repr__(self):
        return f'<User {self.username}>'

class FallRecord(db.Model):
    __tablename__ = 'fall_records'  # 使用原始的 fall_records 表
    id = db.Column(db.Integer, primary_key=True)
    detection_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    lying_down_count = db.Column(db.Integer, nullable=False)
    image_path = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'<FallRecord {self.detection_time}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
    # In lying position, the y-difference between shoulders and hips is small
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

def generate_frames():
    global camera, pose_counts, fall_detected, camera_connected, lying_down_frames
    
    camera = cv2.VideoCapture(0)
    camera_connected = True
    
    while True:
        success, frame = camera.read()
        if not success:
            camera_connected = False
            break
        
        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process pose
        results = pose.process(image)
        
        # Convert back to BGR for OpenCV
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
                    # Save fall image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"fall_{timestamp}.jpg"
                    filepath = os.path.join('static', 'log_photo', filename)
                    cv2.imwrite(filepath, frame)
                    
                    # Save to database
                    fall_record = FallRecord(
                        lying_down_count=pose_counts['lying_down'],
                        image_path=f"log_photo/{filename}"
                    )
                    db.session.add(fall_record)
                    db.session.commit()
            else:
                lying_down_frames = 0
                fall_detected = False
            
            # Display pose type on frame
            cv2.putText(image, f"Pose: {pose_type}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Count: {pose_counts[pose_type]}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if fall_detected:
                cv2.putText(image, "FALL DETECTED!", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

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
    return render_template('dashboard.html')

@app.route('/monitor')
@login_required
def monitor():
    return render_template('monitor.html')

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
        'camera_connected': camera_connected
    })

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
    return render_template('settings.html', users=users)

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
        user = User.query.get(user_id)
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
                'image_path': record.image_path
            } for record in records]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Admin functions (web interface)
@app.route('/settings/reset_pw/<int:user_id>', methods=['POST'])
@login_required
def reset_password(user_id):
    if not current_user.is_admin:
        return redirect(url_for('dashboard'))
    
    user = User.query.get(user_id)
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
    
    user = User.query.get(user_id)
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
    
    user = User.query.get(user_id)
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
    
    record = FallRecord.query.get(record_id)
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
        # 不要使用 create_all()，因為表已經存在
        # db.create_all()
        
        # 檢查是否需要添加 session_token 欄位
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        columns = [c['name'] for c in inspector.get_columns('users')]
        
        if 'session_token' not in columns:
            db.engine.execute('ALTER TABLE users ADD COLUMN session_token VARCHAR(100)')
            print("Added session_token column to users table")
        
        print("Using existing database tables")
        print(f"Existing users: {User.query.count()}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
