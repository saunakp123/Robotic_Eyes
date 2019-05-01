import time
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(1)

while(1):
    ret, img = cap.read()
    img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    height, width, channel = img.shape

    low_range = np.array([100, 100, 100])
    high_range = np.array([120, 255, 255])

    # start = time.time()

    mask = cv.inRange(img_hue, low_range, high_range)

    circles1 = cv.HoughCircles(mask,cv.HOUGH_GRADIENT,1,
    100,param1=100,param2=3,minRadius=3,maxRadius=30)
    circles = circles1[0,:,:]
    circles = np.uint16(np.around(circles))

    if (len(circles) != 4):
        print(len(circles))

    for i in circles:
        cv.circle(img,(i[0],i[1]),20,(0,255,0),3)

    cv.imshow("capture", img)
    cv.imshow('mask', mask)
    # Quit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

