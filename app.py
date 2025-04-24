from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import cv2
import numpy as np
import math
import psycopg2

app = Flask(__name__)
app.secret_key = 'your_secret_key'

camera_connected = False

def get_database_connection():
    return psycopg2.connect(
        dbname="project_database", user="admin",
        password="zxcv1234", host="localhost", port="5432"
    )

print("Loading model...")
model = YOLO("yolo11s-pose.pt")
print("Model loading complete!")

def is_lying_down_advanced(keypoints):
    if keypoints is None or len(keypoints) < 17:
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
        horizontal_torso = abs(angle) < 30 or abs(angle) > 150

        if len(y_values) >= 3:
            x_values = [point[0] for point in key_points]
            y_range = max(y_values) - min(y_values)
            x_range = max(x_values) - min(y_values)
            ratio_check = x_range > y_range * 1.5
            return horizontal_torso and ratio_check

        return horizontal_torso
    except:
        return False

def gen_frames():
    global camera_connected
    conn = get_database_connection()
    cur = conn.cursor()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        camera_connected = False
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "No camera connected"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', img)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    camera_connected = True

    while True:
        success, frame = cap.read()
        if not success:
            camera_connected = False
            break

        results = model(frame)
        standing_sitting_count = 0
        lying_down_count = 0
        annotated_frame = frame.copy()

        for result in results:
            if result.keypoints is not None and hasattr(result.keypoints, 'data'):
                for person_idx in range(len(result.keypoints.data)):
                    keypoints = result.keypoints.data[person_idx].cpu().numpy()
                    if is_lying_down_advanced(keypoints):
                        lying_down_count += 1
                    else:
                        standing_sitting_count += 1
            try:
                annotated_frame = result.plot()
            except:
                pass

        try:
            cur.execute("INSERT INTO pose_counts (standing_sitting, lying_down) VALUES (%s, %s)",
                        (standing_sitting_count, lying_down_count))
            conn.commit()
        except:
            conn.rollback()

        cv2.putText(annotated_frame, f"Standing/Sitting: {standing_sitting_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Lying down: {lying_down_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    cur.close()
    conn.close()

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_status')
def camera_status():
    return jsonify({"connected": camera_connected})

@app.route('/counts')
def counts():
    conn = get_database_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT standing_sitting, lying_down FROM pose_counts ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        return jsonify({
            "standing_sitting": row[0] if row else 0,
            "lying_down": row[1] if row else 0,
            "camera_connected": camera_connected
        })
    except:
        return jsonify({"error": "Unable to fetch pose counts", "camera_connected": camera_connected}), 500
    finally:
        cur.close()
        conn.close()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)
        conn = get_database_connection()
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_pw))
            conn.commit()
            flash('註冊成功，請登入')
            return redirect(url_for('login'))
        except:
            conn.rollback()
            flash('註冊失敗，帳號可能已存在')
        finally:
            cur.close()
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_database_connection()
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user and check_password_hash(user[0], password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('登入失敗，請檢查帳號密碼')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
