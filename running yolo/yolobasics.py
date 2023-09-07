from ultralytics import YOLO
import cv2

#pretrained model(nano,m-median,l-large)--- in relation to camera
model = YOLO('../weights/yolov8n.pt')
results = model("images/img3.jpg", show=True)

cv2.waitKey(0)


