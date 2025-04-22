from ultralytics import YOLO
import cv2

model = YOLO('../yolo_weight/yolov8l.pt')
cv2.namedWindow('Window Name', cv2.WINDOW_NORMAL)

# Resize the window to specific width and height (width, height)
cv2.resizeWindow('Window Name', 50, 50)
results = model("last.jpg", show=True)
cv2.waitKey(0)
