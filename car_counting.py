from ultralytics import YOLO
import cv2
import math
import numpy as np
import cvzone
from sort import *

cap = cv2.VideoCapture("data/cars.mp4")
mask = cv2.imread("data/mask.png")

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCounts = []

model = YOLO("yolov8l.pt")

classNames = ["car"]

while True:
    ret, img = cap.read()
    if not ret:
        break
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))  # 拿到的mask.png跟影片尺寸沒有對其，所以這邊將遮罩縮放到跟影片一樣
    masked = cv2.bitwise_and(img, mask_resized)
    
    imgGraphics = cv2.imread("data/graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    
    results = model(masked, stream=True, classes=[2], device=0)  # COCO資料集定義的第三個(2)是car
    
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)            
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if conf >= 0.5:  # 過濾掉一些可能誤判的物體
                
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'ID: {int(Id)}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        
        
        cx, cy = x1 + w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (0, 0, 255),cv2.FILLED)
        
        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[1]+20:
            if totalCounts.count(Id) == 0:
                totalCounts.append(Id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    cv2.putText(img, str(len(totalCounts)), (255, 100), 
                cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
        
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)