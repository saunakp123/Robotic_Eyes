import cv2 as cv
import numpy as np
import time

# cap = cv.VideoCapture(0)
# while(cap.isOpened()):
# 	#img = cv.imread('D:/Personal/UW/Robotic Eyes/multidots_black.jpg',0)
# 	#height,width = img.shape
# 	ret, img = cap.read()
	
# 	key = cv.waitKey(1) & 0xFF
# 	if key == ord('s'):
# 		cv.imwrite('black.jpg',img)
# 	elif key == ord('c'):
# 		cv.imwrite('bluedot.jpg',img)
# 	elif key == ord('q'):
# 		break
# 	cv.imshow("capture", img)
# cap.release()
# cv.destroyAllWindows()

img1 = cv.imread('test_pictures/cap_white.jpg')
img2 = cv.imread('test_pictures/cap_white_dot.jpg')
gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
lower_limit = np.array([100, 100, 100])
upper_limit = np.array([120, 255, 255])
mask1 = cv.inRange(img1,lower_limit,upper_limit)
mask2 = cv.inRange(img2,lower_limit,upper_limit)
mask_diff = abs(mask2 - mask1)

diff = abs(gray1 - gray2)
# diff = cv.GaussianBlur(diff, (5,5), 0)
cv.imshow('diff', diff)
# cv.imshow('mask2', mask2)

circles1 = cv.HoughCircles(diff,cv.HOUGH_GRADIENT,1,
100,param1=100,param2=30,minRadius=5,maxRadius=30)
circles = circles1[0,:,:]
circles = np.uint16(np.around(circles))

for i in circles:
	cv.circle(img2,(i[0],i[1]),20,(0,255,0),3)

cv.imshow('img',img2)
cv.imwrite('test_results/diff.jpg',diff)

cv.waitKey(0)
cv.destroyAllWindows()

# cv.imwrite('diff.jpg',mask_diff)
# output = cv.connectedComponentsWithStats(mask_diff, 4,cv.CV_32S)
# num_labels = output[0]
# labels = output[1]
# stats = output[2]
# centroid = output[3]
# center = []
# radius = [] 
# for i in range(1,num_labels):
	
# #center.append(i) = [centroid[i][0],centroid[i][1]]
# 	center.append([centroid[i][0], centroid[i][1]])
# 	radius.append([centroid[i][0]-stats[i][0]])
# 	centerfine = np.uint16(np.around(center))
# 	radiusfine = np.uint16(np.around(radius))
# # print(centerfine)
# # print(radiusfine)	
# cv.imshow("capture1", mask_diff)
	