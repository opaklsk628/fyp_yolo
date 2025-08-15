import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from config import Config
from models.database import db, FallRecord
from services.notification_service import NotificationService
from utils.helpers import get_hk_time
import traceback

class DetectionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if self.initialized:
            return
            
        self.model = YOLO('yolo11s-pose.pt')
        
        self.notification_service = NotificationService()
        
        self.person_states = {}
        
        self.pose_counts = {'standing/sitting': 0, 'lying_down': 0}
        
        self.fall_detected = False
        self.fall_alert_active = False
        self.last_fall_record_time = 0
        
        self.debug_mode = False
        
        self.initialized = True
    
    def process_frame(self, frame):


        frame_height = frame.shape[0]
        
        results = self.model(frame, conf=0.5)
        
        vis_frame = frame.copy()
        
        # Reset pose counts
        self.pose_counts = {'standing/sitting': 0, 'lying_down': 0}
        
        current_time = time.time()
        
        detected_persons = []
        
        if len(results) > 0:
            # Check if persons are detected
            if results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Get keypoints data
                keypoints = results[0].keypoints.data if results[0].keypoints else None
                
                # Process each detected person
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [x1, y1, x2, y2]
                    confidence = box.conf[0].cpu().numpy()
                    
                    person_id = self._match_person(bbox, current_time)
                    
                    # ID If new person, create new ID
                    if person_id is None:
                        person_id = f"person_{int(current_time * 1000)}"
                        self.person_states[person_id] = {
                            'lying_start_time': None,
                            'current_pose': 'standing',
                            'last_seen': current_time,
                            'bbox': bbox,
                            'stable_lying': False
                        }
                    
                    detected_persons.append(person_id)
                    
                    self.person_states[person_id]['last_seen'] = current_time
                    self.person_states[person_id]['bbox'] = bbox
                    
                    # Use improved pose detection
                    if keypoints is not None and i < len(keypoints):
                        # Use keypoint method for pose detection
                        pose = self._detect_pose_keypoints(keypoints[i], frame_height)
                    else:
                        # backup plan, use bounding box method
                        pose = self._detect_pose_bbox(bbox, frame_height)
                    
                    self.person_states[person_id]['current_pose'] = pose
                    
                    # Handle different poses
                    if pose == 'lying':
                        # Person is lying
                        if self.person_states[person_id]['lying_start_time'] is None:
                            # Just start lying
                            self.person_states[person_id]['lying_start_time'] = current_time
                            self.person_states[person_id]['stable_lying'] = False
                            
                            if self.debug_mode:
                                print(f"{person_id} Started lying at {current_time:.2f}")
                        
                        # Calculate lying duration
                        lying_duration = current_time - self.person_states[person_id]['lying_start_time']
                        
                        # Update count
                        self.pose_counts['lying_down'] += 1
                        
                        # Red for lying
                        color = (0, 0, 255)
                        status_text = f"Lying: {lying_duration:.1f}s"
                        
                        # Check if fall alert should trigger
                        if lying_duration >= Config.LYING_DOWN_SECONDS:
                            if not self.person_states[person_id]['stable_lying']:
                                self.person_states[person_id]['stable_lying'] = True
                                self._handle_fall_detection(frame)
                        
                        # Draw red bounding box
                        cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(vis_frame, status_text, 
                                  (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, color, 2)
                    
                    else:
                        # standing or sitting, Reset lying timer
                        if self.person_states[person_id]['lying_start_time'] is not None:
                            if self.debug_mode:
                                lying_duration = current_time - self.person_states[person_id]['lying_start_time']
                                print(f"{person_id} sitting up，lying {lying_duration:.1f} sec. Got up after {lying_duration:.1f}s")
                            
                            self.person_states[person_id]['lying_start_time'] = None
                            self.person_states[person_id]['stable_lying'] = False
                        
                        # Update count
                        self.pose_counts['standing/sitting'] += 1
                        
                        # Green for standing or sitting
                        color = (0, 255, 0)
                        
                        # Show correct pose all in green
                        if pose == 'sitting':
                            status_text = "Sitting"
                        else:
                            status_text = "Standing"
                        
                        # Draw green bounding box
                        cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(vis_frame, status_text, 
                                  (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, color, 2)
        
        # Clean up old tracked persons
        self._cleanup_old_persons(detected_persons, current_time)
        
        # Check if need to clear fall alert
        if self.pose_counts['lying_down'] == 0 and self.fall_alert_active:
            # Delay clearing alert
            if current_time - self.last_fall_record_time > 10:
                self.fall_detected = False
                self.fall_alert_active = False
        
        self._add_overlay_info(vis_frame, detected_persons)
        
        return vis_frame
    
    def _add_overlay_info(self, frame, detected_persons):
        """
        Args:
            frame: Video frame
            detected_persons: List of detected persons
        """

        cv2.putText(frame, f"Location: {Config.CAMERA_LOCATION}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        

        cv2.putText(frame, f"Total Persons: {len(detected_persons)}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        

        cv2.putText(frame, f"Standing/Sitting: {self.pose_counts.get('standing/sitting', 0)}", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        

        lying_count = self.pose_counts.get('lying_down', 0)
        cv2.putText(frame, f"Lying Down: {lying_count}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if lying_count > 0 else (0, 255, 0), 2)
        
        # Fall alert
        if self.fall_detected:
            cv2.putText(frame, "FALL DETECTED!", (10, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _detect_pose_keypoints(self, keypoints, frame_height):
        """
        Determine pose standing/sitting/lying based on keypoints
        
        logic:
        1. Torso angle analysis
        2. Ground contact detection
        3.Relative height analysis
        
        Args:
            keypoints: 17 keypoints data (17, 3) - x, y, confidence
            frame_height: Frame height
        Returns:
            str: 'standing', 'sitting', or 'lying'
        """
        try:
            # Convert keypoints format
            kpts = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints
            
            # COCO 17 keypoints index definition
            NOSE = 0
            LEFT_EYE = 1
            RIGHT_EYE = 2
            LEFT_SHOULDER = 5
            RIGHT_SHOULDER = 6
            LEFT_HIP = 11
            RIGHT_HIP = 12
            LEFT_KNEE = 13
            RIGHT_KNEE = 14
            LEFT_ANKLE = 15
            RIGHT_ANKLE = 16
            
            # Minimum confidence threshold
            min_confidence = 0.5
            
            # Estimate ground level
            ground_level = frame_height * 0.9  # Default ground position
            
            ankle_ys = []
            if kpts[LEFT_ANKLE][2] > min_confidence:
                ankle_ys.append(kpts[LEFT_ANKLE][1])
            if kpts[RIGHT_ANKLE][2] > min_confidence:
                ankle_ys.append(kpts[RIGHT_ANKLE][1])
            
            if ankle_ys:
                ground_level = max(ankle_ys)  # Lowest ankle as ground
            
            # Collect key body parts
            shoulders = []
            if kpts[LEFT_SHOULDER][2] > min_confidence:
                shoulders.append(kpts[LEFT_SHOULDER][:2])
            if kpts[RIGHT_SHOULDER][2] > min_confidence:
                shoulders.append(kpts[RIGHT_SHOULDER][:2])
            
            hips = []
            if kpts[LEFT_HIP][2] > min_confidence:
                hips.append(kpts[LEFT_HIP][:2])
            if kpts[RIGHT_HIP][2] > min_confidence:
                hips.append(kpts[RIGHT_HIP][:2])
            
            knees = []
            if kpts[LEFT_KNEE][2] > min_confidence:
                knees.append(kpts[LEFT_KNEE][:2])
            if kpts[RIGHT_KNEE][2] > min_confidence:
                knees.append(kpts[RIGHT_KNEE][:2])
            
            # Check minimum keypoints requirement
            if len(shoulders) == 0 or len(hips) == 0:
                return 'standing'  # Default to standing
            
            # Calculate torso angle
            shoulder_center = np.mean(shoulders, axis=0)
            hip_center = np.mean(hips, axis=0)
            
            # Calculate torso vector
            torso_vector = hip_center - shoulder_center
            torso_dx = torso_vector[0]
            torso_dy = torso_vector[1]
            
            # Calculate angle with vertical line
            torso_angle = np.abs(np.arctan2(torso_dx, np.abs(torso_dy)) * 180 / np.pi)
            
            if self.debug_mode:
                print(f"Torso angle: {torso_angle:.1f}°")
            
            # Calculate height from ground
            shoulder_height = ground_level - shoulder_center[1]
            hip_height = ground_level - hip_center[1]
            
            # Hip height percentage from ground
            hip_height_ratio = hip_height / frame_height
            
            if self.debug_mode:
                print(f"Hip height ratio: {hip_height_ratio:.2f}")
            
            # Comprehensive Judgment
            
            # Lying Detection
            # Condition 1: Torso nearly horizontal
            # Condition 2: Hip close to ground
            if torso_angle > 60 and hip_height_ratio < 0.15:
                # Additional check: shoulders also close to ground
                shoulder_height_ratio = shoulder_height / frame_height
                if shoulder_height_ratio < 0.2:
                    if self.debug_mode:
                        print(f">>> Detected LYING")
                    return 'lying'
            
            # Sitting Detection
            # Condition 1: Hip at medium height
            # Condition 2: Torso relatively vertical
            if 0.15 <= hip_height_ratio <= 0.4:
                if torso_angle < 45:
                    # Check if knees are bent
                    if knees and len(knees) > 0:
                        knee_y = np.mean([k[1] for k in knees])
                        if knee_y > hip_center[1]:  # Knees below hips
                            if self.debug_mode:
                                print(f">>> Detected SITTING")
                            return 'sitting'
                    # Even without knee data, medium hip height with vertical torso might be sitting
                    elif torso_angle < 30:
                        if self.debug_mode:
                            print(f">>> Detected SITTING (no knee)")
                        return 'sitting'
            
            # Standing Detection
            # Condition 1: Hip at high position
            # Condition 2: Torso vertical
            if hip_height_ratio > 0.35 and torso_angle < 30:
                if self.debug_mode:
                    print(f">>>Detected STANDING")
                return 'standing'
            
            # Additional: head and ankle relative position
            head_y = None
            if kpts[NOSE][2] > min_confidence:
                head_y = kpts[NOSE][1]
            elif kpts[LEFT_EYE][2] > min_confidence and kpts[RIGHT_EYE][2] > min_confidence:
                head_y = (kpts[LEFT_EYE][1] + kpts[RIGHT_EYE][1]) / 2
            
            if head_y is not None and ankle_ys:
                head_ankle_diff = abs(head_y - np.mean(ankle_ys))
                if head_ankle_diff < frame_height * 0.2:
                    if self.debug_mode:
                        print(f">>>Detected LYING (head-ankle)")
                    return 'lying'
            
            # Use body vertical range
            all_visible_y = []
            for i in [0, 5, 6, 11, 12, 13, 14, 15, 16]:
                if kpts[i][2] > min_confidence:
                    all_visible_y.append(kpts[i][1])
            
            if len(all_visible_y) >= 4:
                body_vertical_range = max(all_visible_y) - min(all_visible_y)
                body_range_ratio = body_vertical_range / frame_height
                
                if body_range_ratio < 0.15:
                    if self.debug_mode:
                        print(f">>>Detected LYING (body range)")
                    return 'lying'
                elif body_range_ratio > 0.4:
                    if self.debug_mode:
                        print(f">>>Detected STANDING (body range)")
                    return 'standing'
            
            # Default judgment
            if torso_angle > 45:
                return 'lying'
            elif hip_height_ratio < 0.3:
                return 'sitting'
            else:
                return 'standing'
            
        except Exception as e:
            print(f"Pose analysis error: {e}")
            return 'standing'
    
    def _detect_pose_bbox(self, bbox, frame_height):
        """
        backup plan, determine pose based on bounding box
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            frame_height: Frame height
        Returns:
            str: 'standing', 'sitting', or 'lying'
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        aspect_ratio = width / height if height > 0 else 0
        
        # Calculate distance from bbox bottom to frame bottom
        bottom_distance = frame_height - y2
        bottom_ratio = bottom_distance / frame_height
        
        # Calculate bbox center height
        center_y = (y1 + y2) / 2
        center_height_ratio = (frame_height - center_y) / frame_height
        
        if self.debug_mode:
            print(f"BBox method: ={aspect_ratio:.2f}, ={center_height_ratio:.2f}")
        
        # Judgment logic
        if aspect_ratio > 1.8 and bottom_ratio < 0.1:
            return 'lying'
        elif 0.8 < aspect_ratio < 1.5 and 0.2 < center_height_ratio < 0.5:
            return 'sitting'
        else:
            return 'standing'
    
    def _match_person(self, bbox, current_time):
        best_match = None
        best_iou = 0.3  # Lower threshold for better matching
        
        for person_id, state in self.person_states.items():
            if current_time - state['last_seen'] > Config.MAX_TRACKING_AGE:
                continue
                
            iou = self._calculate_iou(bbox, state['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match = person_id
        
        return best_match
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _cleanup_old_persons(self, detected_persons, current_time):
        """
        Remove persons not seen recently
        
        Args:
            detected_persons: Persons detected in current frame
            current_time: Current time
        """
        persons_to_remove = []
        
        for person_id, state in self.person_states.items():
            if person_id not in detected_persons:
                # Keep lying persons longer
                if state['current_pose'] == 'lying':
                    max_age = Config.MAX_TRACKING_AGE * 3
                else:
                    max_age = Config.MAX_TRACKING_AGE
                
                if current_time - state['last_seen'] > max_age:
                    persons_to_remove.append(person_id)
                    
                    if self.debug_mode and state['current_pose'] == 'lying':
                        print(f"Removing lying person {person_id}")
        
        for person_id in persons_to_remove:
            del self.person_states[person_id]
    
    def _handle_fall_detection(self, frame):
        """
        Handle fall detection event
        Save image, record to database, send notification
        
        Args:
            frame: Current video frame
        """
        current_fall_time = time.time()
        
        # Check cooldown to avoid duplicate records
        if current_fall_time - self.last_fall_record_time >= Config.FALL_RECORD_COOLDOWN:
            self.fall_detected = True
            self.fall_alert_active = True
            self.last_fall_record_time = current_fall_time
            
            print("Fall detected! Saving record...")
            
            hk_time = get_hk_time()
            timestamp = hk_time.strftime("%Y%m%d_%H%M%S")
            filename = f"fall_{timestamp}.jpg"
            filepath = os.path.join('static', 'log_photo', filename)
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            save_success = cv2.imwrite(filepath, frame)
            print(f"Image save {'successful' if save_success else 'failed'}: {filepath}")
            
            # Save to database and send notification
            try:
                from app import create_app
                app = create_app()
                with app.app_context():
                    fall_record = FallRecord(
                        lying_down_count=self.pose_counts.get('lying_down', 1),
                        image_path=f"log_photo/{filename}",
                        camera_location=Config.CAMERA_LOCATION
                    )
                    db.session.add(fall_record)
                    db.session.commit()
                    print(f"DEBUG: Database record saved. ID: {fall_record.id}")
                    
                    # Send Telegram notification
                    hk_time_str = hk_time.strftime('%Y-%m-%d %H:%M:%S')
                    message = (
                        "🚨 <b>FALL DETECTED!</b> 🚨\n"
                        f"⏰ Time: {hk_time_str}\n"
                        f"📍 Location: {Config.CAMERA_LOCATION}\n"
                        f"👥 Persons lying down: {self.pose_counts.get('lying_down', 1)}\n"
                        "⚠️ <b>IMMEDIATE ATTENTION REQUIRED!</b>"
                    )
                    
                    notification_sent = self.notification_service.send_telegram_notification(message, filepath)
                    if notification_sent:
                        print("Telegram Notification sent successfully")
                    else:
                        print("DEBUG: Telegram Notification failed")
                    
            except Exception as e:
                print(f"ERROR saving fall record: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                try:
                    db.session.rollback()
                except:
                    pass
    
    def get_counts(self):

        from services.camera_service import CameraService
        camera_service = CameraService()
        
        max_lying_seconds = 0
        current_time = time.time()
        
        if self.person_states:
            for person_id, state in self.person_states.items():
                if state['current_pose'] == 'lying' and state['lying_start_time']:
                    lying_seconds = current_time - state['lying_start_time']
                    max_lying_seconds = max(max_lying_seconds, lying_seconds)
        
        return {
            'standing_sitting': self.pose_counts.get('standing/sitting', 0),
            'lying_down': self.pose_counts.get('lying_down', 0),
            'fall_detected': self.fall_detected,
            'camera_connected': camera_service.camera_connected,
            'current_resolution': camera_service.current_resolution,
            'camera_location': Config.CAMERA_LOCATION,
            'lying_seconds': round(max_lying_seconds, 1),
            'lying_progress': min(100, int((max_lying_seconds / Config.LYING_DOWN_SECONDS) * 100)),
            'total_persons': len(self.person_states)
        }
    
    def set_debug_mode(self, enabled=True):

        self.debug_mode = enabled
        print(f"Debug mode: {'ON' if enabled else 'OFF'}")



