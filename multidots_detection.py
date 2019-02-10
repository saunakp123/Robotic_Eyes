import numpy as np
import cv2 as cv
 
img = cv.imread("test_pictures/multidots_black.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# MinDist: Minimum distance between the centers of the detected circles.
# Param2: The smaller it is, the more false circles may be detected.
# MinRadius/MaxRadius: Define the radius range of circles that can be detected (in pixels)
circles1 = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,
minDist=100,param1=100,param2=10,minRadius=10,maxRadius=100)

circles = circles1[0,:,:]
circles = np.uint16(np.around(circles))

print(len(circles))
for i in circles: 
	cv.circle(img,(i[0],i[1]),20,(0,255,0),3)
	print(i)

cv.namedWindow('test', cv.WINDOW_NORMAL)
cv.imshow('test',img)
# cv.imwrite('test_results/multi_calibration_dots.jpg',img)

cv.waitKey(0)
cv.destroyAllWindows()