from ultralytics import YOLO
import cv2
import numpy as np

yolo = YOLO('../weights/yolov8n.pt')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)  # Adjust the resolution as needed

while True:
    success, img = cap.read()

    if not success:
        break

    cv2.imshow("image", img)
    cv2.waitKey(1)


