from ultralytics import YOLO
import cv2
import numpy as np
import math
import cvzone
from sort import *

cap = cv2.VideoCapture("../videos/cars.mp4")
cap.set(3, 640)
cap.set(4, 480)  # Adjust the resolution as needed

model = YOLO('../weights/yolov8n.pt')

classNames = [
    "person", "cell phone", "suitcase", "hair", "refrigerator", "toothbrush",
    "pen", "motorbike", "pencil", "bus", "car", "dog", "cat",
    "chair", "table", "bottle", "book", "laptop",
    "keyboard", "mouse", "phone", "cup", "plate",
    "fork", "knife", "spoon", "watch", "hat",
    "shoe", "shirt", "pants", "dress", "glasses",
    "hat", "bag", "backpack", "umbrella", "guitar",
    "hat", "cupcake", "flower", "tree", "house",
    "boat", "airplane", "train", "truck", "bus", "motorcycle",
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

mask = cv2.imread("../CarCounter/mask.png")

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [300, 550, 400, 350]
totalCount = []
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
while True:
    success, img = cap.read()

    if not success:
        break

    imgregion = cv2.bitwise_and(img, mask)

    results = model(imgregion, stream=True)
    # print(results)
    detections = np.empty((0, 5))

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

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # class names
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if (currentclass == "car" or currentclass == "truck" or currentclass == "train" or currentclass == "motorbike") and confidence > 0.45:
                # cvzone.putTextRect(img, f"{currentclass}{confidence}", (max(35, x1), max(35, y1)), scale=1, thickness=1,
                #                    offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentarray = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentarray))

    results_for_tracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in results_for_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5, colorR=(255, 0, 0))

        cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(0, y1)), scale=1, thickness=1, offset=3)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20: # 20 detection time not to detect too early
        #     if totalCount.count(id) == 0:
        #         totalCount.append(id)

        if id not in totalCount:
            totalCount.append(id)

    cvzone.putTextRect(img, f" Count: {len(totalCount)}", (50, 50))

    cv2.imshow("image", img)
    # cv2.imshow("imgregion", imgregion)
    cv2.waitKey(1)

