from ultralytics import YOLO
import cv2
import numpy as np
import math

# Loading pre-trained model
print("Loading model...")
model = YOLO("yolo11n-pose.pt")
print("Model loading complete!")

def is_lying_down(keypoints):
    """
    Analyze body keypoints to determine if the person is in a lying down position
    keypoints: Keypoint coordinates detected by the model
    """
# Check if keypoints array is empty or insufficient
    if keypoints is None or len(keypoints) < 17:
        print("Insufficient number of keypoints")
        return False
        
# Extract shoulder and hip keypoints
    try:
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
    except IndexError:
        print("Unable to retrieve required keypoints")
        return False
    
# Check if keypoints are valid
    if all(point[2] > 0.5 for point in [left_shoulder, right_shoulder, left_hip, right_hip]):
        # Calculate torso vector angle
        torso_vector_x = ((left_hip[0] + right_hip[0]) / 2) - ((left_shoulder[0] + right_shoulder[0]) / 2)
        torso_vector_y = ((left_hip[1] + right_hip[1]) / 2) - ((left_shoulder[1] + right_shoulder[1]) / 2)
        
        # Calculate angle with horizontal line
        angle = math.degrees(math.atan2(torso_vector_y, torso_vector_x))
        print(f"Detected angle: {angle:.2f} degrees")
        
        # If torso is close to horizontal (angle between -30 to 30 or 150 to -150), classify as lying down
        lying_down = abs(angle) < 30 or abs(angle) > 150
        return lying_down
    
    print("Unable to detect complete keypoints")
    return False

# Process a single image
def process_image(image_path):
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Unable to read image: {image_path}")
        return
    
    height, width = img.shape[:2]
    print(f"Image resolution: {width}x{height}")
    
    results = model(img)
    
# Display results on the image
    for result in results:
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            keypoints = result.keypoints.data[0].cpu().numpy()  # Take the first detected person
            
            lying_down = is_lying_down(keypoints)
            posture = "Lying down" if lying_down else "Standing/Sitting"
            print(f"Detected posture: {posture}")
            
            # Display posture label on the image
            cv2.putText(img, f"Posture: {posture}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            annotated_frame = result.plot()
            cv2.imshow("Pose Detection", annotated_frame)
        else:
            print("No body or keypoints detected")
            cv2.imshow("Pose Detection", img)
    
    print("Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process video stream
def process_video(video_source=0, width=1920, height=1080):  # 0 represents default camera
    print(f"Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Unable to open video source: {video_source}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")
    
    print("Video processing started, press 'Q' to exit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Unable to read video frame")
            break
            
        results = model(frame)
        
        annotated_frame = frame.copy()
        
        for result in results:
            if (result.keypoints is not None and 
                hasattr(result.keypoints, 'data') and 
                len(result.keypoints.data) > 0 and 
                len(result.keypoints.data[0]) > 0):
                
                try:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    lying_down = is_lying_down(keypoints)
                    posture = "Lying down" if lying_down else "Standing/Sitting"
                    

                    cv2.putText(frame, f"Posture: {posture}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error processing keypoints: {e}")
            
            try:
                annotated_frame = result.plot()
            except Exception as e:
                print(f"Error drawing results: {e}")
                annotated_frame = frame
        
        cv2.imshow("Pose Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User pressed 'q' key, exiting")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing ended")

if __name__ == "__main__":
    import sys
    import os
    
    try:
        if len(sys.argv) > 1:
            # If image path is provided as a parameter
            image_path = sys.argv[1]
            if os.path.exists(image_path):
                process_image(image_path)
            else:
                print(f"Image not found: {image_path}")
        else:
            process_video(width=1920, height=1080)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    print("Program execution complete")
