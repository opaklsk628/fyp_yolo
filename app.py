from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
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

# log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

camera_connected = False
current_standing_sitting = 0
current_lying_down = 0
fall_detected = False  # 全局變數標記是否偵測到跌倒

def get_database_connection():
    try:
        return psycopg2.connect(
            dbname="project_database", user="admin",
            password="zxcv1234", host="localhost", port="5432"
        )
    except Exception as e:
        logger.error(f"數據庫連接失敗: {e}")
        return None

def try_create_tables():
    conn = get_database_connection()
    if not conn:
        logger.error("無法連接到數據庫，無法啟動")
        return False
    
    cur = conn.cursor()
    tables_created = True
    
    try:
        # 建立 users 表（如果不存在）
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL
            )
        ''')
        
        # 檢查 fall_records 表是否存在
        cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'fall_records')")
        table_exists = cur.fetchone()[0]
        
        if table_exists:
            # 檢查是否有 timestamp 列（舊版）或 detection_time 列（新版）
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'fall_records'")
            columns = [col[0] for col in cur.fetchall()]
            
            if 'timestamp' in columns and 'detection_time' not in columns:
                # 如果存在 timestamp 但不存在 detection_time，則需要修復表結構
                logger.info("檢測到舊版表結構，正在更新...")
                
                # 備份原有數據
                cur.execute("CREATE TEMP TABLE fall_records_backup AS SELECT * FROM fall_records")
                
                # 刪除舊表並創建新表
                cur.execute("DROP TABLE fall_records")
                cur.execute('''
                    CREATE TABLE fall_records (
                        id SERIAL PRIMARY KEY,
                        detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        lying_down_count INTEGER DEFAULT 0,
                        image_path VARCHAR(255) NOT NULL
                    )
                ''')
                
                # 恢復數據（將 timestamp 列的數據遷移到 detection_time 列）
                cur.execute('''
                    INSERT INTO fall_records (id, detection_time, lying_down_count, image_path)
                    SELECT id, timestamp, lying_down_count, image_path FROM fall_records_backup
                ''')
                
                logger.info("表結構已更新，原有數據已遷移")
            elif 'timestamp' in columns and 'detection_time' in columns:
                # 如果兩列都存在，可能是遷移過程中出錯，需要清理
                logger.warning("檢測到重複列，正在修復...")
                cur.execute("ALTER TABLE fall_records DROP COLUMN timestamp")
                logger.info("已移除多餘的 timestamp 列")
            elif 'detection_time' not in columns:
                # 如果表存在但缺乏 detection_time 列（其他情況），添加它
                cur.execute("ALTER TABLE fall_records ADD COLUMN detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                logger.info("已添加缺少的 detection_time 列")
            else:
                logger.info("fall_records 表結構正確，無需修改")
        else:
            # 表不存在，創建新表
            cur.execute('''
                CREATE TABLE IF NOT EXISTS fall_records (
                    id SERIAL PRIMARY KEY,
                    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    lying_down_count INTEGER DEFAULT 0,
                    image_path VARCHAR(255) NOT NULL
                )
            ''')
            logger.info("已創建 fall_records 表")
        
        conn.commit()
        logger.info("數據庫表格初始化成功")
    except Exception as e:
        conn.rollback()
        logger.error(f"建立或更新表格錯誤: {e}")
        tables_created = False
    finally:
        cur.close()
        conn.close()
    
    return tables_created

print("Loading model...")
model = YOLO("yolo11s-pose.pt")
print("Model loading complete")

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
            x_range = max(x_values) - min(x_values)
            ratio_check = x_range > y_range * 1.5
            return horizontal_torso and ratio_check

        return horizontal_torso
    except:
        return False

def save_fall_record(image_path, frame, lying_down_count):
    """
    在獨立的線程中保存跌倒記錄
    為每個保存操作創建新的資料庫連接
    """
    try:
        # 保存圖片
        cv2.imwrite(image_path, frame)
        
        # 獲取新的數據庫連接
        conn = get_database_connection()
        if not conn:
            logger.error("無法保存跌倒記錄：無法連接到資料庫")
            return
            
        cur = conn.cursor()
        try:
            # 只保存檔案名，非完整路徑
            filename = os.path.basename(image_path)
            file_path = f"log_photo/{filename}"  # 相對於 static 目錄的路徑
            
            cur.execute("INSERT INTO fall_records (image_path, lying_down_count) VALUES (%s, %s)", 
                       (file_path, lying_down_count))
            conn.commit()
            logger.info(f"成功保存跌倒記錄，躺下人數: {lying_down_count}，圖片: {file_path}")
        except Exception as e:
            conn.rollback()
            logger.error(f"保存跌倒記錄失敗: {e}")
        finally:
            # 確保關閉連接
            cur.close()
            conn.close()
    except Exception as e:
        logger.error(f"處理跌倒記錄時發生錯誤: {e}")

def gen_frames():
    global camera_connected, current_standing_sitting, current_lying_down, fall_detected
    
    if not os.path.exists('static/log_photo'):
        os.makedirs('static/log_photo')
    
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
    
    # 防止太頻繁保存
    last_save_time = 0
    save_interval = 3  # 至少間隔3秒才保存新記錄

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

        current_standing_sitting = standing_sitting_count
        current_lying_down = lying_down_count
        logger.info(f"Updated counts: standing_sitting={standing_sitting_count}, lying_down={lying_down_count}")

        # 檢查是否偵測到跌倒且間隔時間足夠
        current_time = time.time()
        if lying_down_count > 0 and (current_time - last_save_time) >= save_interval:
            fall_detected = True
            last_save_time = current_time
            image_filename = f"fall_{int(current_time)}.jpg"
            image_path = os.path.join('static/log_photo', image_filename)
            
            # 使用修改後的 save_fall_record 函數，不再傳入連接
            Thread(target=save_fall_record, args=(image_path, frame.copy(), lying_down_count)).start()
        else:
            fall_detected = (lying_down_count > 0)

        cv2.putText(annotated_frame, f"Standing/Sitting: {standing_sitting_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Lying down: {lying_down_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

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
    global fall_detected
    logger.info(f"Returning counts: standing_sitting={current_standing_sitting}, lying_down={current_lying_down}, fall_detected={fall_detected}")
    return jsonify({
        "standing_sitting": current_standing_sitting,
        "lying_down": current_lying_down,
        "fall_detected": fall_detected,
        "camera_connected": camera_connected
    })

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)
        
        conn = get_database_connection()
        if not conn:
            flash('無法連接到資料庫，請稍後再試')
            return render_template('register.html')
        
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_pw))
            conn.commit()
            flash('註冊成功，請登入')
            return redirect(url_for('login'))
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            flash('註冊失敗，帳號已存在')
            return render_template('register.html')
        except Exception as e:
            conn.rollback()
            logger.error(f"註冊時資料庫錯誤: {e}")
            flash('註冊時發生錯誤，請稍後再試')
            return render_template('register.html')
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
        if not conn:
            flash('無法連接到資料庫，請稍後再試')
            return render_template('login.html')
        
        cur = conn.cursor()
        try:
            cur.execute("SELECT username, password_hash FROM users WHERE username = %s", (username,))
            user = cur.fetchone()
            
            if user and check_password_hash(user[1], password):
                session['username'] = username
                return redirect(url_for('index'))
            else:
                flash('登入失敗，請檢查帳號密碼')
        except Exception as e:
            logger.error(f"登入時資料庫錯誤: {e}")
            flash('登入時發生錯誤，請稍後再試')
        finally:
            cur.close()
            conn.close()
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/fall_records')
def fall_records():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    conn = get_database_connection()
    if not conn:
        flash('無法連接到資料庫，無法檢視跌倒記錄')
        return render_template('fall_records.html', records=[])
    
    records = []
    cur = conn.cursor()
    try:
        cur.execute("SELECT detection_time, lying_down_count, image_path FROM fall_records ORDER BY detection_time DESC")
        db_records = cur.fetchall()
        logger.info(f"從資料庫獲取到 {len(db_records)} 條跌倒記錄")
        
        # 處理每個記錄，確保圖片路徑正確
        for record in db_records:
            detection_time = record[0]
            lying_count = record[1]
            image_path = record[2]
            
            # 確保路徑格式一致
            if image_path.startswith('static/'):
                # 從完整路徑中提取檔案名
                image_file = image_path.split('/')[-1]
                path_for_template = f"log_photo/{image_file}"
            elif image_path.startswith('log_photo/'):
                # 已經是相對路徑，保持不變
                path_for_template = image_path
            else:
                # 其他情況，假設是純檔案名
                path_for_template = f"log_photo/{image_path}"
            
            # 檢查檔案是否存在
            full_path = os.path.join('static', path_for_template)
            if os.path.exists(full_path):
                records.append((detection_time, lying_count, path_for_template))
            else:
                logger.warning(f"跌倒記錄圖片不存在: {full_path}")
                # 仍然添加記錄，但路徑可能顯示不出圖片
                records.append((detection_time, lying_count, path_for_template))
        
        logger.info(f"處理後得到 {len(records)} 條有效跌倒記錄")
    except Exception as e:
        logger.error(f"獲取跌倒記錄失敗: {e}")
        flash('獲取跌倒記錄時發生錯誤')
    finally:
        cur.close()
        conn.close()
    
    return render_template('fall_records.html', records=records)

# 在應用啟動時檢查數據庫連接並初始化表格
def init_app():
    # 嚴格檢查數據庫連接
    if not get_database_connection():
        logger.critical("無法連接到數據庫，應用程式將退出")
        print("錯誤：無法連接到數據庫，請檢查數據庫設定")
        sys.exit(1)  # 終止運作
    
    # 建立表格
    if not try_create_tables():
        logger.critical("無法創建必要的數據庫表格，應用程式將退出")
        print("錯誤：無法創建必要的數據庫表格，請檢查數據庫權限")
        sys.exit(1)  # 終止應用程式
    
    logger.info("數據庫連接正常，表格初始化成功")
    print("數據庫初始化成功，應用程式已準備好")


init_app()

if __name__ == '__main__':
    app.run(debug=True)
