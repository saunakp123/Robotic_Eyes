import time
import numpy as np
import cv2 as cv

img = cv.imread("test_pictures/1.jpg")
img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# opencv index use height*width
height, width, channel = img.shape

low_range = np.array([156, 43, 46])
high_range = np.array([180, 255, 255])

start = time.time()

mask = cv.inRange(img_hue, low_range, high_range)
points = cv.findNonZero(mask)
# average these points
avg = np.mean(points, axis=0)
avg = avg[0]
coord = (avg[0]/height, avg[1]/width)

center = (int(avg[0]), int(avg[1]))

end = time.time()
print(end - start)
# print(count)
cv.circle(img, center, 5, (0,255,0), -1)
cv.namedWindow('test', cv.WINDOW_NORMAL)
cv.imshow('test', img)

cv.waitKey(0)
cv.destroyAllWindows()
