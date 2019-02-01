import time
import numpy as np
import cv2 as cv

img = cv.imread("test_pictures/1.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
# opencv index use height*width
sp = img.shape

count = 0

start = time.time()

for y in range(0, sp[0]):
	for x in range(0, sp[1]):
		# color in B G R order
		color = img[y, x]
		if color[2] > 254:
			# count = count + 1
			img[y, x] = [0, 0, 0]

end = time.time()
print(end - start)
# print(count)
cv.namedWindow('test', cv.WINDOW_NORMAL)
cv.imshow('test', img)

cv.waitKey(0)
cv.destroyAllWindows()
