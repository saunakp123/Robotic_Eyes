import time
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while(1):
    ret, img = cap.read()
    img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    height, width, channel = img.shape

    low_range = np.array([156, 43, 46])
    high_range = np.array([180, 255, 255])

    # start = time.time()

    mask = cv.inRange(img_hue, low_range, high_range)
    points = cv.findNonZero(mask)
    if (points is None):
    	center = ((0, 0))
    else:	
	    # Average these points
	    avg = np.mean(points, axis=0)
	    avg = avg[0]
	    coord = (avg[0]/height, avg[1]/width)

	    center = (int(avg[0]), int(avg[1]))

    # end = time.time()
    # print(end - start)
    # print(count)
    cv.circle(img, center, 5, (0,255,0), -1)

    cv.imshow("capture", img)
    # Quit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

