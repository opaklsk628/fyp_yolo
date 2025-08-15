import sys
import pytz
from datetime import datetime

HK_TZ = pytz.timezone('Asia/Hong_Kong')
UTC = pytz.UTC

def get_hk_time():
    """Get current time in Hong Kong timezone"""
    return datetime.now(HK_TZ)

def utc_to_hk(utc_dt):
    """Convert UTC datetime to Hong Kong timezone"""
    if utc_dt.tzinfo is None:
        utc_dt = UTC.localize(utc_dt)
    return utc_dt.astimezone(HK_TZ)

def hk_to_utc(hk_dt):
    """Convert Hong Kong datetime to UTC for storage"""
    if hk_dt.tzinfo is None:
        hk_dt = HK_TZ.localize(hk_dt)
    return hk_dt.astimezone(UTC).replace(tzinfo=None)

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print("\nShutting down gracefully...")
    
    # Stop camera service
    from services.camera_service import CameraService
    camera_service = CameraService()
    camera_service.stop()
    
    sys.exit(0)
