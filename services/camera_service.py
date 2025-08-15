import cv2
import threading
import time
import queue
import numpy as np
from config import Config

class CameraService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if self.initialized:
            return
            
        self.camera = None
        self.camera_lock = threading.Lock()
        self.camera_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.latest_frame = None
        self.latest_processed_frame = None
        self.camera_connected = False
        self.background_processing = True
        self.current_resolution = Config.DEFAULT_RESOLUTION
        self.restart_camera_flag = False
        self.initialized = True
    
    def start(self):
        """Start camera processing thread"""
        if self.camera_thread is None or not self.camera_thread.is_alive():
            self.background_processing = True
            self.camera_thread = threading.Thread(target=self._process_camera_background, daemon=True)
            self.camera_thread.start()
            print("Camera service started")
    
    def stop(self):
        """Stop camera processing"""
        self.background_processing = False
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)
        print("Camera service stopped")
    
    def restart(self):
        """Restart camera service"""
        self.stop()
        time.sleep(2)
        self.start()
    
    def set_resolution(self, resolution):
        """Change camera resolution"""
        if resolution in Config.RESOLUTION_SETTINGS:
            self.current_resolution = resolution
            self.restart_camera_flag = True
            return True
        return False
    
    def generate_frames(self):
        """Generate frames for video streaming"""
        quality_settings = {
            '240p': 30, '360p': 40, '480p': 50, '720p': 60, '1080p': 70
        }
        
        while True:
            if self.latest_processed_frame is None:
                width, height = Config.RESOLUTION_SETTINGS[self.current_resolution]
                blank = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera not connected", (width//4, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                quality = quality_settings.get(self.current_resolution, 50)
                ret, buffer = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, quality])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)
                continue
            
            with self.camera_lock:
                frame = self.latest_processed_frame.copy()
            
            quality = quality_settings.get(self.current_resolution, 50)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            fps = Config.FPS_SETTINGS[self.current_resolution]
            time.sleep(1.0 / fps)
    
    def _process_camera_background(self):
        """Background thread for camera processing"""
        # Import detection service here to avoid circular import
        from services.detection_service import DetectionService
        detection_service = DetectionService()
        
        print("Starting background camera processing...")
        retry_count = 0
        max_retries = 5
        
        while self.background_processing:
            try:
                if self.restart_camera_flag:
                    self.restart_camera_flag = False
                    self._restart_camera()
                    continue
                
                if self.camera is None:
                    with self.camera_lock:
                        print(f"Opening camera... (Attempt {retry_count + 1}/{max_retries})")
                        self.camera = cv2.VideoCapture(0)
                        
                        width, height = Config.RESOLUTION_SETTINGS[self.current_resolution]
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        
                        fps = Config.FPS_SETTINGS[self.current_resolution]
                        self.camera.set(cv2.CAP_PROP_FPS, fps)
                        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        if not self.camera.isOpened():
                            print("Failed to open camera")
                            self.camera_connected = False
                            self.camera = None
                            retry_count += 1
                            
                            if retry_count >= max_retries:
                                print(f"Failed after {max_retries} attempts. Waiting 30 seconds...")
                                time.sleep(30)
                                retry_count = 0
                            else:
                                time.sleep(5)
                            continue
                        else:
                            print(f"Camera opened successfully - FPS: {fps}")
                            self.camera_connected = True
                            retry_count = 0
                
                success = False
                frame = None
                
                with self.camera_lock:
                    if self.camera is not None:
                        success, frame = self.camera.read()
                
                if not success or frame is None:
                    print("Failed to read frame")
                    self.camera_connected = False
                    with self.camera_lock:
                        if self.camera is not None:
                            self.camera.release()
                            self.camera = None
                    time.sleep(1)
                    continue
                
                # Resize if needed
                height, width = frame.shape[:2]
                target_width, target_height = Config.RESOLUTION_SETTINGS[self.current_resolution]
                if width != target_width or height != target_height:
                    frame = cv2.resize(frame, (target_width, target_height))
                
                with self.camera_lock:
                    self.latest_frame = frame.copy()
                
                # Process frame with detection service
                processed_frame = detection_service.process_frame(frame)
                
                with self.camera_lock:
                    self.latest_processed_frame = processed_frame
                
                # Sleep based on FPS
                fps = Config.FPS_SETTINGS[self.current_resolution]
                time.sleep(1.0 / fps)
                
            except Exception as e:
                print(f"Error in camera processing: {e}")
                with self.camera_lock:
                    if self.camera is not None:
                        self.camera.release()
                        self.camera = None
                self.camera_connected = False
                time.sleep(5)
        
        print("Camera processing stopped")
    
    def _restart_camera(self):
        """Restart camera with new settings"""
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            
            time.sleep(0.5)
            
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                width, height = Config.RESOLUTION_SETTINGS[self.current_resolution]
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                fps = Config.FPS_SETTINGS[self.current_resolution]
                self.camera.set(cv2.CAP_PROP_FPS, fps)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                print(f"Camera restarted: {self.current_resolution}, FPS: {fps}")
                return True
        return False


