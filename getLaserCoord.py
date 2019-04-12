import cv2 as cv
import numpy as np

def getLaserCoord(img):
	height, width, channel = img.shape

	low_range = np.array([150, 100, 220])
	high_range = np.array([250, 200, 255])

	mask = cv.inRange(img, low_range, high_range)
	points = cv.findNonZero(mask)

	if (points is None):
		return 0
	else:
		# Average these points
		avg = np.mean(points, axis=0)
		avg = avg[0]
		coord = (avg[0]/height, avg[1]/width)

		center = (int(avg[0]), int(avg[1]))

	cv.circle(img, center, 5, (0,255,0), -1)

	return center