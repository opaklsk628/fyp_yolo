from ultralytics import YOLO
import cv2
import numpy as np
import math

model = YOLO("yolo11n-pose.pt")

def is_lying_down(keypoints):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    if all(point[2] > 0.5 for point in [left_shoulder, right_shoulder, left_hip, right_hip]):
        torso_vector_x = ((left_hip[0] + right_hip[0]) / 2) - ((left_shoulder[0] + right_shoulder[0]) / 2)
        torso_vector_y = ((left_hip[1] + right_hip[1]) / 2) - ((left_shoulder[1] + right_shoulder[1]) / 2)
        
        angle = math.degrees(math.atan2(torso_vector_y, torso_vector_x))
        
        return abs(angle) < 30 or abs(angle) > 150
    
    return False

def process_image(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.data[0].cpu().numpy()
            
            lying_down = is_lying_down(keypoints)
            posture = "躺臥" if lying_down else "站立/坐姿"
            
            cv2.putText(img, f"姿勢: {posture}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            result.plot()
    
    cv2.imshow("Pose Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_source=0):
    cap = cv2.VideoCapture(video_source)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        results = model(frame)
        
        for result in results:
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                keypoints = result.keypoints.data[0].cpu().numpy()
                
                lying_down = is_lying_down(keypoints)
                posture = "躺臥" if lying_down else "站立/坐姿"
                
                cv2.putText(frame, f"姿勢: {posture}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        result.plot()
        cv2.imshow("Pose Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
