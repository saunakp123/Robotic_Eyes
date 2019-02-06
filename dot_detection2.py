import time
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while(1):
	# get a frame
	ret, img = cap.read()
	fps = cap.get(5)
	print(fps)
	# opencv index use height*width
	height, width, channel = img.shape

	low_range = np.array([150, 100, 220])
	high_range = np.array([250, 200, 255])

	# start = time.time()

	mask = cv.inRange(img, low_range, high_range)
	points = cv.findNonZero(mask)
	if (points is None):
		center = (0, 0)
	else:	
		# average these points
		avg = np.mean(points, axis=0)
		avg = avg[0]
		coord = (avg[0]/height, avg[1]/width)

		center = (int(avg[0]), int(avg[1]))
		

	# end = time.time()
	# print(end - start)
	# print(count)
	cv.circle(img, center, 5, (0,255,0), -1)

	# show a frame
	cv.imshow("capture", img)
	cv.imshow("mask", mask)
	# quit if 'q' is pressed
	if cv.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv.destroyAllWindows()

