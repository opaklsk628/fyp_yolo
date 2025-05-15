from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, Response, jsonify, abort
)
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import cv2
import numpy as np
import math
import psycopg2
import logging
import os
import time
import sys
from threading import Thread
from functools import wraps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

camera_connected = False
current_standing_sitting = 0
current_lying_down = 0
fall_detected = False  # 全局變數標記是否偵測到跌倒

# database
def get_database_connection():
    """回傳 PostgreSQL 連線（失敗回 None）。"""
    try:
        return psycopg2.connect(
            dbname="project_database",
            user="admin",
            password="zxcv1234",
            host="localhost",
            port="5432"
        )
    except Exception as e:
        logger.error(f"資料庫連線失敗: {e}")
        return None


def try_create_tables():
    conn = get_database_connection()
    if not conn:
        return False

    cur = conn.cursor()
    try:
        # users
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                is_admin BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')

        # fall_records
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fall_records (
                id SERIAL PRIMARY KEY,
                detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                lying_down_count INTEGER DEFAULT 0,
                image_path VARCHAR(255) NOT NULL
            );
        """)

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"初始化表格錯誤: {e}")
        return False
    finally:
        cur.close()
        conn.close()

    return True

print("Loading model…")
model = YOLO("yolo11s-pose.pt")
print("Model loaded")

# defined admin
def admin_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('is_admin'):
            abort(403)
        return f(*args, **kwargs)
    return wrapped

# 判斷人體骨架是否躺下
def is_lying_down_advanced(keypoints):
    if keypoints is None or len(keypoints) < 17:
        return False
    try:
        nose           = keypoints[0]
        left_shoulder  = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip       = keypoints[11]
        right_hip      = keypoints[12]
        left_ankle     = keypoints[15]
        right_ankle    = keypoints[16]

        key_points = [left_shoulder, right_shoulder, left_hip, right_hip]
        if not all(p[2] > 0.5 for p in key_points):
            return False

        y_vals = [p[1] for p in
                  [nose, left_shoulder, right_shoulder,
                   left_hip, right_hip, left_ankle, right_ankle]
                  if p[2] > 0.5]

        shoulders_mid = [(left_shoulder[0] + right_shoulder[0]) / 2,
                         (left_shoulder[1] + right_shoulder[1]) / 2]
        hips_mid = [(left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2]

        torso_vec = [hips_mid[0] - shoulders_mid[0],
                     hips_mid[1] - shoulders_mid[1]]
        angle = math.degrees(math.atan2(torso_vec[1], torso_vec[0]))
        horizontal = abs(angle) < 30 or abs(angle) > 150

        if len(y_vals) >= 3:
            x_vals = [p[0] for p in key_points]
            y_range = max(y_vals) - min(y_vals)
            x_range = max(x_vals) - min(x_vals)
            return horizontal and (x_range > y_range * 1.5)
        return horizontal
    except:
        return False

# 儲存跌倒紀錄, cap圖+寫入database
def save_fall_record(image_path, frame, lying_down_count):
    try:
        cv2.imwrite(image_path, frame)

        conn = get_database_connection()
        if not conn:
            logger.error("無法保存跌倒記錄,連線失敗")
            return
        cur = conn.cursor()
        try:
            filename = os.path.basename(image_path)
            rel_path = f"log_photo/{filename}"
            cur.execute(
                "INSERT INTO fall_records (image_path, lying_down_count) "
                "VALUES (%s, %s)",
                (rel_path, lying_down_count)
            )
            conn.commit()
            logger.info(f"已保存跌倒記錄 ({lying_down_count}) {rel_path}")
        except Exception as e:
            conn.rollback()
            logger.error(f"保存跌倒記錄失敗: {e}")
        finally:
            cur.close(); conn.close()
    except Exception as e:
        logger.error(f"save_fall_record 例外: {e}")

# camera設定
def gen_frames():
    global camera_connected, current_standing_sitting, current_lying_down, fall_detected

    if not os.path.exists('static/log_photo'):
        os.makedirs('static/log_photo')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        camera_connected = False
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            img, "No camera connected",
            (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        _, buf = cv2.imencode('.jpg', img)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')
        return

    camera_connected = True
	
    # 防止太頻繁保存
    last_save_time = 0
    save_interval  = 3  # 至少間隔3秒才保存新記錄

    while True:
        ok, frame = cap.read()
        if not ok:
            camera_connected = False
            break

        results = model(frame)
        standing_sitting = 0
        lying_down       = 0
        annotated = frame.copy()

        for res in results:
            if res.keypoints is not None and hasattr(res.keypoints, 'data'):
                for idx in range(len(res.keypoints.data)):
                    kpt = res.keypoints.data[idx].cpu().numpy()
                    if is_lying_down_advanced(kpt):
                        lying_down += 1
                    else:
                        standing_sitting += 1
            try:
                annotated = res.plot()
            except:
                pass

        current_standing_sitting = standing_sitting
        current_lying_down       = lying_down

        now = time.time()
        if lying_down > 0 and (now - last_save_time) >= save_interval:
            fall_detected = True
            last_save_time = now
            fname = f"fall_{int(now)}.jpg"
            path  = os.path.join('static/log_photo', fname)
            Thread(target=save_fall_record,
                   args=(path, frame.copy(), lying_down)).start()
        else:
            fall_detected = (lying_down > 0)

        cv2.putText(annotated, f"Standing/Sitting: {standing_sitting}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Lying down: {lying_down}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buf = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

    cap.release()

# webpage route
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/counts')
def counts():
    return jsonify({
        "standing_sitting": current_standing_sitting,
        "lying_down": current_lying_down,
        "fall_detected": fall_detected,
        "camera_connected": camera_connected
    })

# register account
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        h_pw     = generate_password_hash(password)

        conn = get_database_connection()
        if not conn:
            flash('資料庫錯誤，稍後再試')
            return render_template('register.html')

        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, h_pw)
            )
            conn.commit()
            flash('註冊完成，請登入')
            return redirect(url_for('login'))
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            flash('帳號已存在')
        except Exception as e:
            conn.rollback()
            logger.error(f"註冊錯誤: {e}")
            flash('發生錯誤，稍後再試')
        finally:
            cur.close(); conn.close()

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_database_connection()
        if not conn:
            flash('資料庫錯誤，稍後再試')
            return render_template('login.html')

        cur = conn.cursor()
        cur.execute(
            "SELECT id, password_hash, is_admin FROM users WHERE username=%s",
            (username,)
        )
        row = cur.fetchone()
        cur.close(); conn.close()

        if row and check_password_hash(row[1], password):
            session['user_id']  = row[0]
            session['username'] = username
            session['is_admin'] = row[2]
            return redirect(url_for('index'))
        flash('帳號或密碼錯誤')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# admin setting page
@app.route('/settings')
@admin_required
def settings():
    conn = get_database_connection()
    cur  = conn.cursor()
    cur.execute("SELECT id, username, is_admin FROM users ORDER BY id")
    users = cur.fetchall()
    cur.close(); conn.close()
    return render_template('settings.html', users=users)


@app.route('/settings/reset_pw/<int:uid>', methods=['POST'])
@admin_required
def reset_pw(uid):
    new_hash = generate_password_hash('123456')
    conn = get_database_connection()
    cur  = conn.cursor()
    cur.execute("UPDATE users SET password_hash=%s WHERE id=%s",
                (new_hash, uid))
    conn.commit(); cur.close(); conn.close()
    flash('密碼已重設為123456')
    return redirect(url_for('settings'))


@app.route('/settings/toggle_admin/<int:uid>', methods=['POST'])
@admin_required
def toggle_admin(uid):
    conn = get_database_connection()
    cur  = conn.cursor()
    cur.execute("UPDATE users SET is_admin = NOT is_admin WHERE id=%s", (uid,))
    conn.commit(); cur.close(); conn.close()
    flash('權限已更改')
    return redirect(url_for('settings'))


@app.route('/settings/delete/<int:uid>', methods=['POST'])
@admin_required
def delete_user(uid):
    if uid == session.get('user_id'):
        flash('不能刪除自己')
        return redirect(url_for('settings'))

    conn = get_database_connection()
    cur  = conn.cursor()
    cur.execute("DELETE FROM users WHERE id=%s", (uid,))
    conn.commit(); cur.close(); conn.close()
    flash('帳號已刪除')
    return redirect(url_for('settings'))

# 跌倒記錄
@app.route('/fall_records')
def fall_records():
    if 'username' not in session:
        return redirect(url_for('login'))

    conn = get_database_connection()
    if not conn:
        flash('無法連線資料庫')
        return render_template('fall_records.html', records=[])

    records = []
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT detection_time, lying_down_count, image_path "
            "FROM fall_records ORDER BY detection_time DESC"
        )
        for dt, cnt, img in cur.fetchall():
            if img.startswith('static/'):
                img = img.split('/')[-1]
            if not img.startswith('log_photo/'):
                img = f"log_photo/{img}"
            records.append((dt, cnt, img))
    except Exception as e:
        logger.error(f"讀取跌倒記錄失敗: {e}")
        flash('讀取記錄失敗')
    finally:
        cur.close(); conn.close()

    return render_template('fall_records.html', records=records)

# 啟動前檢查
def init_app():
    if not get_database_connection():
        logger.critical('資料庫連線失敗，程式終止')
        sys.exit(1)
    if not try_create_tables():
        logger.critical('表格初始化失敗，程式終止')
        sys.exit(1)
    logger.info('資料庫就緒')

init_app()

if __name__ == '__main__':
    app.run(debug=True)

