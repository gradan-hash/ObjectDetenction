from ultralytics import YOLO
import cv2
import numpy as np
import math
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)  # Adjust the resolution as needed

model = YOLO('../weights/yolov8n.pt')
while True:
    success, img = cap.read()

    if not success:
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # for open cv

            # USING CVZONE MUCH ADVANCED AND MORE FEATURES
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("confidence ", confidence)

            cvzone.putTextRect(img, f"{confidence}", (x1, y1 - 20))

    cv2.imshow("image", img)
    cv2.waitKey(1)
