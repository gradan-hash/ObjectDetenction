from ultralytics import YOLO
import cv2
import numpy as np
import math
import cvzone

cap = cv2.VideoCapture(0)  #for webcam
cap.set(3, 640)
cap.set(4, 480)  # Adjust the resolution as needed

model = YOLO('../weights/yolov8n.pt')

classNames = [
    "person", "cell phone", "suitcase", "hair", "refrigerator", "toothbrush",
    "pen", "pencil", "car", "dog", "cat",
    "chair", "table", "bottle", "book", "laptop",
    "keyboard", "mouse", "phone", "cup", "plate",
    "fork", "knife", "spoon", "watch", "hat",
    "shoe", "shirt", "pants", "dress", "glasses",
    "hat", "bag", "backpack", "umbrella", "guitar",
    "hat", "cupcake", "flower", "tree", "house",
    "boat", "airplane", "train", "bus", "motorcycle",
    "bicycle", "ball", "frisbee", "skateboard", "surfboard",
    "kite", "teddy bear", "doll", "sunglasses", "bed",
    "chair", "couch", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book",
    "clock", "vase", "scissors", "teddy bear", "hairbrush",
    "hair dryer", "toothbrush", "toilet", "hair dryer", "sink",
    "pen", "pencil", "marker", "knife", "fork",
    "spoon", "cup", "plate", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair"
]

while True:
    success, img = cap.read()

    if not success:
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # for open cv

            # USING CVZONE MUCH ADVANCED AND MORE FEATURES
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # class names
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f"{classNames[cls]}{confidence}", (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("image", img)
    cv2.waitKey(1)
