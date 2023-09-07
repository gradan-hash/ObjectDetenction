from ultralytics import YOLO
import cv2
model = YOLO('../weights/yolov8n.pt')
results = model("images/img3.jpg",show=True)

cv2.waitKey(0)


