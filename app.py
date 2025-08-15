#!/usr/bin/env python3
"""
Heatstroke Detection System - Main Application
"""
import signal
import sys
from flask import Flask
from flask_login import LoginManager
from models.database import db, User
from config import Config
from routes import auth_bp, main_bp, api_bp, admin_bp
from services.camera_service import CameraService
from utils.helpers import signal_handler

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize extensions
    db.init_app(app)
    
    # Initialize login manager
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    
    return app

def init_database(app):
    """Initialize database with default data"""
    with app.app_context():
        db.create_all()
        
        # Create default admin if no users exist
        if User.query.count() == 0:
            from werkzeug.security import generate_password_hash
            admin = User(
                username='admin',
                password_hash=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created: admin/admin123")
        
        # Initialize system settings
        from models.database import SystemSettings
        if SystemSettings.query.count() == 0:
            settings = SystemSettings()
            db.session.add(settings)
            db.session.commit()
            print("Default system settings created")

if __name__ == '__main__':
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create application
    app = create_app()
    
    # Initialize database
    init_database(app)
    
    # Start camera service
    camera_service = CameraService()
    camera_service.start()
    
    # Run application
    print("Starting Heatstroke Detection System...")
    print("Access the system at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
