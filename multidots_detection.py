import numpy as np
import cv2 as cv
 
img = cv.imread("test_pictures/multidots.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
 
circles1 = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,
100,param1=100,param2=30,minRadius=10,maxRadius=100)

circles = circles1[0,:,:]
circles = np.uint16(np.around(circles))

print(len(circles))
for i in circles: 
	cv.circle(img,(i[0],i[1]),20,(0,255,0),3)
	print(i)

cv.namedWindow('test', cv.WINDOW_NORMAL)
cv.imshow('test',img)
cv.waitKey(0)
cv.destroyAllWindows()