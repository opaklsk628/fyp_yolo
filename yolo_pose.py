from ultralytics import YOLO
import cv2
import numpy as np
import math

print("Loading model...")
model = YOLO("yolo11n-pose.pt")
print("Model loading complete!")

def is_lying_down_advanced(keypoints):
    """
    Advanced analysis of body keypoints to determine if the person is lying down
    using multiple criteria including torso angle and keypoint distribution
    """
# Check if keypoints array is empty or insufficient
    if keypoints is None or len(keypoints) < 17:
        print("Insufficient number of keypoints")
        return False
        
    try:
# Extract key body landmarks
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
# Check confidence for essential keypoints
        key_points = [left_shoulder, right_shoulder, left_hip, right_hip]
        if not all(point[2] > 0.5 for point in key_points):
            print("Low confidence in essential keypoints")
            return False
        
# Calculate vertical distribution of key points with sufficient confidence
        y_values = [point[1] for point in [nose, left_shoulder, right_shoulder, 
                                          left_hip, right_hip, left_ankle, right_ankle] 
                    if point[2] > 0.5]
        
# Calculate midpoints of shoulders and hips
        shoulders_midpoint = [(left_shoulder[0] + right_shoulder[0])/2, 
                              (left_shoulder[1] + right_shoulder[1])/2]
        hips_midpoint = [(left_hip[0] + right_hip[0])/2, 
                         (left_hip[1] + right_hip[1])/2]
        
# Calculate torso vector angle with horizontal
        torso_vector = [hips_midpoint[0] - shoulders_midpoint[0], 
                        hips_midpoint[1] - shoulders_midpoint[1]]
        angle = math.degrees(math.atan2(torso_vector[1], torso_vector[0]))
        print(f"Detected torso angle: {angle:.2f} degrees")
        
# Criterion 1: Torso is approximately horizontal
        horizontal_torso = abs(angle) < 30 or abs(angle) > 150
        
# Criterion 2: Analyze distribution of keypoints in horizontal vs vertical space
        if len(y_values) >= 3:
# Get x coordinates of key points
            x_values = [point[0] for point in key_points]
            
# Calculate ranges and standard deviation
            y_range = max(y_values) - min(y_values)
            y_std = np.std(y_values)
            x_range = max(x_values) - min(x_values)
            
# When lying down, horizontal spread should be greater than vertical spread
            ratio_check = x_range > y_range * 1.5
            
            print(f"X-range: {x_range:.2f}, Y-range: {y_range:.2f}, Ratio check: {ratio_check}")
            
# Combined criteria for improved accuracy
            return horizontal_torso and ratio_check
        
# Fall back to just torso angle if we don't have enough points for distribution analysis
        return horizontal_torso
        
    except Exception as e:
        print(f"Error in posture analysis: {e}")
        return False

def process_image(image_path):
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Unable to read image: {image_path}")
        return
    
    height, width = img.shape[:2]
    print(f"Image resolution: {width}x{height}")
    
    results = model(img)
    
    standing_sitting_count = 0
    lying_down_count = 0
    
    for result in results:
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            # Process all detected people
            for person_idx in range(len(result.keypoints.data)):
                keypoints = result.keypoints.data[person_idx].cpu().numpy()
                
                # Use the advanced lying down detection
                lying_down = is_lying_down_advanced(keypoints)
                
                # Update counters
                if lying_down:
                    lying_down_count += 1
                else:
                    standing_sitting_count += 1
            
            # Display count information on the image
            cv2.putText(img, f"Standing/Sitting: {standing_sitting_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Lying down: {lying_down_count}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw keypoints and skeleton
            annotated_frame = result.plot()
            cv2.imshow("Pose Detection", annotated_frame)
        else:
            print("No body or keypoints detected")
            cv2.putText(img, "Standing/Sitting: 0", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, "Lying down: 0", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Pose Detection", img)
    
    print(f"Detection results - Standing/Sitting: {standing_sitting_count}, Lying down: {lying_down_count}")
    print("Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_source=0, width=1920, height=1080):
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
        
        standing_sitting_count = 0
        lying_down_count = 0
        
        annotated_frame = frame.copy()
        
        for result in results:
            # Check if people were detected
            if (result.keypoints is not None and 
                hasattr(result.keypoints, 'data') and 
                len(result.keypoints.data) > 0):
                
                # Process each detected person
                for person_idx in range(len(result.keypoints.data)):
                    try:
                        keypoints = result.keypoints.data[person_idx].cpu().numpy()
                        
                        # Use the advanced lying down detection method
                        lying_down = is_lying_down_advanced(keypoints)
                        
                        # Update counters
                        if lying_down:
                            lying_down_count += 1
                        else:
                            standing_sitting_count += 1
                            
                    except Exception as e:
                        print(f"Error processing keypoints for person {person_idx}: {e}")
            
            # Draw keypoints and skeleton, even if posture analysis fails
            try:
                annotated_frame = result.plot()
            except Exception as e:
                print(f"Error drawing results: {e}")
                # If drawing fails, use original frame
                annotated_frame = frame
        
        # Display count information on the image
        cv2.putText(annotated_frame, f"Standing/Sitting: {standing_sitting_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Lying down: {lying_down_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame with detection results
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
