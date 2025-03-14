from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import math
import redis

app = Flask(__name__)

redis_client = redis.Redis(host='localhost', port=6379, db=0)

print("Loading model...")
model = YOLO("yolo11s-pose.pt")
print("Model loading complete!")

def is_lying_down_advanced(keypoints):
    if keypoints is None or len(keypoints) < 17:
        print("Insufficient number of keypoints")
        return False
    try:
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]

        key_points = [left_shoulder, right_shoulder, left_hip, right_hip]
        if not all(point[2] > 0.5 for point in key_points):
            print("Low confidence in essential keypoints")
            return False

        y_values = [point[1] for point in [nose, left_shoulder, right_shoulder, 
                                             left_hip, right_hip, left_ankle, right_ankle]
                    if point[2] > 0.5]

        shoulders_midpoint = [(left_shoulder[0] + right_shoulder[0]) / 2, 
                              (left_shoulder[1] + right_shoulder[1]) / 2]
        hips_midpoint = [(left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2]

        torso_vector = [hips_midpoint[0] - shoulders_midpoint[0],
                        hips_midpoint[1] - shoulders_midpoint[1]]
        angle = math.degrees(math.atan2(torso_vector[1], torso_vector[0]))
        print(f"Detected torso angle: {angle:.2f} degrees")

        horizontal_torso = abs(angle) < 30 or abs(angle) > 150

        if len(y_values) >= 3:
            x_values = [point[0] for point in key_points]
            y_range = max(y_values) - min(y_values)
            x_range = max(x_values) - min(x_values)
            ratio_check = x_range > y_range * 1.5
            print(f"X-range: {x_range:.2f}, Y-range: {y_range:.2f}, Ratio check: {ratio_check}")
            return horizontal_torso and ratio_check

        return horizontal_torso

    except Exception as e:
        print(f"Error in posture analysis: {e}")
        return False

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open video source")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        standing_sitting_count = 0
        lying_down_count = 0
        annotated_frame = frame.copy()

        for result in results:
            if (result.keypoints is not None and 
                hasattr(result.keypoints, 'data') and 
                len(result.keypoints.data) > 0):
                for person_idx in range(len(result.keypoints.data)):
                    try:
                        keypoints = result.keypoints.data[person_idx].cpu().numpy()
                        lying_down = is_lying_down_advanced(keypoints)
                        if lying_down:
                            lying_down_count += 1
                        else:
                            standing_sitting_count += 1
                    except Exception as e:
                        print(f"Error processing keypoints for person {person_idx}: {e}")

            try:
                annotated_frame = result.plot()
            except Exception as e:
                print(f"Error drawing results: {e}")
                annotated_frame = frame

        redis_client.set('standing_sitting', standing_sitting_count)
        redis_client.set('lying_down', lying_down_count)

        cv2.putText(annotated_frame, f"Standing/Sitting: {standing_sitting_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Lying down: {lying_down_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def counts():
    standing_sitting = redis_client.get('standing_sitting')
    lying_down = redis_client.get('lying_down')
    counts_data = {
        "standing_sitting": int(standing_sitting.decode()) if standing_sitting else 0,
        "lying_down": int(lying_down.decode()) if lying_down else 0
    }
    return jsonify(counts_data)

if __name__ == '__main__':
    app.run(debug=True)
