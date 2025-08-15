from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from models.database import db, User, FallRecord, SystemSettings
from services.weather_service import WeatherService
from services.notification_service import NotificationService
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from sqlalchemy import func
from utils.helpers import get_hk_time, hk_to_utc
import uuid
import pytz
import os

api_bp = Blueprint('api', __name__)
HK_TZ = pytz.timezone('Asia/Hong_Kong')

weather_service = WeatherService()
notification_service = NotificationService()

@api_bp.route('/login', methods=['POST'])
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

@api_bp.route('/register', methods=['POST'])
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
        
        new_user = User(
            username=username,
            password_hash=generate_password_hash(password),
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

@api_bp.route('/weather')
def api_weather():
    weather = weather_service.get_weather()
    if weather:
        return jsonify({'success': True, 'data': weather})
    else:
        return jsonify({'success': False, 'message': 'Failed to fetch weather data'}), 500

@api_bp.route('/statistics')
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
        
        total_count = db.session.query(func.count(FallRecord.id)).scalar() or 0
        
        return jsonify({
            'success': True,
            'statistics': {
                'today': today_count,
                'this_month': month_count,
                'this_year': year_count,
                'total': total_count
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@api_bp.route('/recent_falls')
def api_recent_falls():
    """Get recent fall records for live updates"""
    try:
        from config import Config
        records = FallRecord.query.order_by(FallRecord.detection_time.desc()).limit(5).all()
        return jsonify({
            'success': True,
            'falls': [{
                'id': record.id,
                'time': record.hk_detection_time.strftime('%H:%M:%S'),
                'date': record.hk_detection_time.strftime('%Y-%m-%d'),
                'lying_down_count': record.lying_down_count,
                'camera_location': record.camera_location or Config.CAMERA_LOCATION
            } for record in records]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'falls': []}), 200

@api_bp.route('/fall_records', methods=['GET'])
def api_fall_records():
    """Get fall records in JSON format"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'success': False, 'message': 'No authorization token'}), 401
    
    try:
        from config import Config
        records = FallRecord.query.order_by(FallRecord.detection_time.desc()).all()
        return jsonify({
            'success': True,
            'records': [{
                'id': record.id,
                'detection_time': record.hk_detection_time.strftime('%Y-%m-%d %H:%M:%S'),
                'lying_down_count': record.lying_down_count,
                'image_path': record.image_path,
                'camera_location': record.camera_location or Config.CAMERA_LOCATION
            } for record in records]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@api_bp.route('/fall_records/<int:record_id>', methods=['DELETE'])
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

@api_bp.route('/users', methods=['GET'])
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
                'id': u.id,
                'username': u.username,
                'is_admin': u.is_admin
            } for u in users]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@api_bp.route('/users/<int:user_id>/toggle_admin', methods=['POST'])
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

@api_bp.route('/telegram_settings', methods=['GET', 'POST'])
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
                notification_service.send_telegram_notification(test_message)
            
            return jsonify({'success': True, 'message': 'Settings updated successfully'})
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': str(e)}), 500

@api_bp.route('/resolution', methods=['GET', 'POST'])
def api_resolution():
    """Get or set resolution for mobile app"""
    from services.camera_service import CameraService
    camera_service = CameraService()
    
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'success': False, 'message': 'No authorization token'}), 401
    
    token = auth_header.split(' ')[1]
    user = User.query.filter_by(session_token=token).first()
    
    if not user:
        return jsonify({'success': False, 'message': 'Invalid token'}), 401
    
    if request.method == 'GET':
        from config import Config
        return jsonify({
            'success': True,
            'current_resolution': camera_service.current_resolution,
            'available_resolutions': list(Config.RESOLUTION_SETTINGS.keys())
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        new_resolution = data.get('resolution', '720p')
        
        if camera_service.set_resolution(new_resolution):
            return jsonify({'success': True, 'resolution': new_resolution})
        else:
            return jsonify({'success': False, 'message': 'Invalid resolution'}), 400

@api_bp.route('/system/restart', methods=['POST'])
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
        from services.camera_service import CameraService
        camera_service = CameraService()
        camera_service.restart()
        
        return jsonify({'success': True, 'message': 'System restarted'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@api_bp.route('/system/stop', methods=['POST'])
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
        import os
        import signal
        
        print("System stop requested...")
        os.kill(os.getpid(), signal.SIGTERM)
        
        return jsonify({'success': True, 'message': 'System shutdown initiated'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
