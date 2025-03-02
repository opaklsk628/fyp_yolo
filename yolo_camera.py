import cv2
from ultralytics import YOLO

model = YOLO("your_trained_model.pt")  # 使用你訓練好的模型

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can not detect camera")
        break

    results = model(frame)

    for result in results:
        annotated_frame = result.plot()
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name == "lying_down":  # 檢查是否為橫向或躺下的姿勢
                cv2.putText(annotated_frame, "Lying Down", (int(box.xyxy[0][0]), int(box.xyxy[0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
