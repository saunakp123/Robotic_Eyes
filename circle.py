import time
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while(1):
    ret, img = cap.read()
    img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    height, width, channel = img.shape

    low_range = np.array([, 50, 50])
    high_range = np.array([255, 150, 150])

    # start = time.time()

    mask = cv.inRange(img_hue, low_range, high_range)

    cv.imshow("capture", img)
    cv.imshow('mask', mask)
    # Quit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

