from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user
from models.database import db, User, FallRecord, SystemSettings
from werkzeug.security import generate_password_hash
from services.notification_service import NotificationService
import os

admin_bp = Blueprint('admin', __name__)
notification_service = NotificationService()

@admin_bp.route('/settings')
@login_required
def settings():
    if not current_user.is_admin:
        flash('Admin access required')
        return redirect(url_for('main.dashboard'))
    
    users = User.query.all()
    telegram_settings = SystemSettings.query.first()
    return render_template('settings.html', users=users, telegram_settings=telegram_settings)

@admin_bp.route('/settings/telegram', methods=['POST'])
@login_required
def update_telegram_settings():
    if not current_user.is_admin:
        return redirect(url_for('main.dashboard'))
    
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
        if notification_service.send_telegram_notification(test_message):
            flash('Telegram settings saved and test message sent!')
        else:
            flash('Telegram settings saved but test message failed. Please check your settings.')
    else:
        flash('Telegram settings saved!')
    
    return redirect(url_for('admin.settings'))

@admin_bp.route('/settings/reset_pw/<int:user_id>', methods=['POST'])
@login_required
def reset_password(user_id):
    if not current_user.is_admin:
        return redirect(url_for('main.dashboard'))
    
    user = db.session.get(User, user_id)
    if user:
        new_password = '123456'
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        flash(f'Password reset for {user.username}. New password is: {new_password}')
    
    return redirect(url_for('admin.settings'))

@admin_bp.route('/settings/toggle_admin/<int:user_id>', methods=['POST'])
@login_required
def toggle_admin(user_id):
    if not current_user.is_admin:
        return redirect(url_for('main.dashboard'))
    
    user = db.session.get(User, user_id)
    if user and user.id != current_user.id:
        user.is_admin = not user.is_admin
        db.session.commit()
        flash(f'Admin status toggled for {user.username}')
    
    return redirect(url_for('admin.settings'))

@admin_bp.route('/settings/delete/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        return redirect(url_for('main.dashboard'))
    
    user = db.session.get(User, user_id)
    if user and user.id != current_user.id:
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.username} deleted')
    
    return redirect(url_for('admin.settings'))

@admin_bp.route('/fall_records/delete/<int:record_id>', methods=['POST'])
@login_required
def delete_fall_record(record_id):
    if not current_user.is_admin:
        return redirect(url_for('main.fall_records'))
    
    record = db.session.get(FallRecord, record_id)
    if record:
        image_path = os.path.join('static', record.image_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        db.session.delete(record)
        db.session.commit()
        flash('Record deleted')
    
    return redirect(url_for('main.fall_records'))

@admin_bp.route('/test_fall_detection', methods=['POST'])
@login_required
def test_fall_detection():
    """Manually trigger a fall detection for testing"""
    if not current_user.is_admin:
        flash('Admin access required')
        return redirect(url_for('main.dashboard'))
    
    try:
        import cv2
        import numpy as np
        from utils.helpers import get_hk_time
        from config import Config
        
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
            image_path=f"log_photo/{filename}",
            camera_location=Config.CAMERA_LOCATION
        )
        db.session.add(fall_record)
        db.session.commit()
        
        hk_time_str = hk_time.strftime('%Y-%m-%d %H:%M:%S')
        message = (
            "🧪 <b>TEST: Fall Detection</b>\n"
            f"Time: {hk_time_str}\n"
            f"Location: {Config.CAMERA_LOCATION}\n"
            "This is a test notification."
        )
        
        notification_service.send_telegram_notification(message, filepath)
        
        flash('Test fall detection created successfully')
        return redirect(url_for('main.fall_records'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}')
        return redirect(url_for('main.dashboard'))
