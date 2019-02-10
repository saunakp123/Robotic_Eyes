import time
import numpy as np
import cv2 as cv

img = cv.imread("test_pictures/1.jpg")
# OpenCV index use height*width
sp = img.shape

count = 0

start = time.time()

for y in range(0, sp[0]):
	for x in range(0, sp[1]):
		# Color in B G R order
		color = img[y, x]
		if color[2] > 254:
			print(color)
			# count = count + 1
			img[y, x] = [0, 0, 0]

end = time.time()
print(end - start)
# print(count)

# The displayed window can be resized
cv.namedWindow('test', cv.WINDOW_NORMAL)
cv.imshow('test', img)

cv.waitKey(0)
cv.destroyAllWindows()
