import cv2 as cv
import numpy as np
import datetime

from getLaserCoord import getLaserCoord
from getWarp import getWarp

SAVE_VIDEO = False
flag = False

cap = cv.VideoCapture(1)
current = datetime.datetime.now()
current = current.strftime("%H-%M-%S")

RATIO = 1.73

COL = 480
ROW = COL*RATIO
ret, img = cap.read()
M = getWarp(img, RATIO)

input("press any key to continue: ")

while(cap.isOpened()):

	ret, img = cap.read()
	dst = cv.warpPerspective(img,M,(int(ROW),COL))
	center = getLaserCoord(dst)
	# center = getLaserCoord(img)
	print(center)

	cv.imshow('img', img)
	cv.imshow('dst', dst)
	if cv.waitKey(1) & 0xFF == ord('q'):
	    break

cap.release()
cv.destroyAllWindows()
