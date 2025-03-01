import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

#connect to camera, 0=default
cap = cv2.VideoCapture(0)

#camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can not detect camera")
        break

    #use yolo model to detect object
    results = model(frame)

    for result in results:
        annotated_frame = result.plot()  #show the object name in output video

        cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    #pass q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
