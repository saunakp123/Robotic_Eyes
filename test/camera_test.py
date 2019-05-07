import numpy as np
import cv2 as cv

# The index is for different cameras
cap = cv.VideoCapture(0)

while(cap.isOpened()):
    # Get a frame
    ret, frame = cap.read()
    # Show a frame
    cv.imshow("capture", cv.flip(frame, 180))
    # Quit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Remember to release the camera at the end
cap.release()
cv.destroyAllWindows()
