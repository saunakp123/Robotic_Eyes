import numpy as np
import cv2 as cv

cap = cv.VideoCapture(1)
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv.imshow("capture", cv.flip(frame, 180))
    # quit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
master