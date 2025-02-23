import cv2 as cv

#img = cv.imread('baki_pf.jpg')

#cv.imshow('baki', img)

capture = cv.VideoCapture('last of us.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('video', frame)

    if cv.waitKey(20) & 0xFF== ord('d'):
        break

capture.release()
cv.destroyAllWindows()

#cv.waitKey(0)
