from flask import Blueprint, render_template, redirect, url_for, Response, request, jsonify
from flask_login import login_required, current_user
from services.camera_service import CameraService
from services.weather_service import WeatherService
from services.detection_service import DetectionService
from models.database import db, FallRecord
from utils.helpers import get_hk_time, hk_to_utc
from datetime import datetime, timedelta
from sqlalchemy import func
import pytz

main_bp = Blueprint('main', __name__)
HK_TZ = pytz.timezone('Asia/Hong_Kong')

# Service instances
camera_service = CameraService()
weather_service = WeatherService()
detection_service = DetectionService()

@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('auth.login'))

@main_bp.route('/dashboard')
@login_required
def dashboard():
    weather = weather_service.get_weather()
    
    # Calculate statistics
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

@main_bp.route('/monitor')
@login_required
def monitor():
    from config import Config
    weather = weather_service.get_weather()
    recent_falls = FallRecord.query.order_by(FallRecord.detection_time.desc()).limit(5).all()
    return render_template('monitor.html', 
                         weather=weather,
                         current_resolution=camera_service.current_resolution,
                         recent_falls=recent_falls,
                         camera_location=Config.CAMERA_LOCATION)

@main_bp.route('/video_feed')
def video_feed():
    return Response(camera_service.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@main_bp.route('/counts')
def get_counts():
    return jsonify(detection_service.get_counts())

@main_bp.route('/set_resolution', methods=['POST'])
@login_required
def set_resolution():
    data = request.get_json()
    new_resolution = data.get('resolution', '720p')
    success = camera_service.set_resolution(new_resolution)
    
    if success:
        return jsonify({'success': True, 'resolution': new_resolution})
    else:
        return jsonify({'success': False, 'message': 'Invalid resolution'}), 400

@main_bp.route('/fall_records')
@login_required
def fall_records():
    records = FallRecord.query.order_by(FallRecord.detection_time.desc()).all()
    return render_template('fall_records.html', records=records)
