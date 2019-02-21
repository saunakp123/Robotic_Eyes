import cv2 as cv
import numpy as np
import time
cap = cv.VideoCapture(0)
start = time.clock()
print(start)
while(cap.isOpened()):
	#img = cv.imread('D:/Personal/UW/Robotic Eyes/multidots_black.jpg',0)
	#height,width = img.shape
	ret, img = cap.read()
	
	key = cv.waitKey(1) & 0xFF
	if key == ord('s'):
		cv.imwrite('D:/Personal/UW/Robotic/black.jpg',img)
	elif key == ord('c'):
		cv.imwrite('D:/Personal/UW/Robotic/bluedot.jpg',img)

img1 = cv.imread('D:/Personal/UW/Robotic Eyes/black.jpg')
img2 = cv.imread('D:/Personal/UW/Robotic Eyes/bluedot.jpg')
lower_limit = np.array([100, 100, 100])
upper_limit = np.array([120, 255, 255])
hsv_mask1 = cv.inRange(img1,lower_limit,upper_limit)
hsv_mask2 = cv.inRange(img2,lower_limit,upper_limit)
mask_diff = hsv_mask2 - hsv_mask1 
output = cv.connectedComponentsWithStats(img, 4,cv.CV_32S)
num_labels = output[0]
labels = output[1]
stats = output[2]
centroid = output[3]

for i in range(1,num_labels):
	
#center.append(i) = [centroid[i][0],centroid[i][1]]
	center.append([centroid[i][0], centroid[i][1]])
	radius.append([centroid[i][0]-stats[i][0]])
	centerfine = np.uint16(np.around(center))
	radiusfine = np.uint16(np.around(radius))
print(centerfine)
print(radiusfine)	

	