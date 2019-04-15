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
M = getWarp(img)

input("press any key to continue: ")

while(cap.isOpened()):

	ret, img = cap.read()
	dst = cv.warpPerspective(img,M,(int(ROW),COL))
	center = getLaserCoord(dst)

cap.release()
cv.destroyAllWindows()
