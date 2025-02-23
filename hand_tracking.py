import cv2
import mediapipe as mp
import time

wCam, hCam = 640, 480
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(3, hCam)


mpHands = mp.solutions.hands
hands = mpHands.Hands()
pTime = 0

while True:
    success, img = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)


    cv2.imshow("image", img)
    cv2.waitKey(1)
