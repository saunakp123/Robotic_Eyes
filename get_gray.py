import time
import numpy as np
import cv2 as cv

img = cv.imread("test_pictures/1.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
# OpenCV index use height*width
sp = img_gray.shape

count = 0

start = time.time()

for y in range(0, sp[0]):
	for x in range(0, sp[1]):
		intensity = img_gray[y, x]
		if intensity > 205:
			# count = count + 1
			# print(intensity)
			img_gray[y, x] = 0

end = time.time()
print(end - start)

print(count)

cv.namedWindow('2', cv.WINDOW_NORMAL)
cv.imshow('2', img_gray)
cv.waitKey(0)
cv.destroyAllWindows()
