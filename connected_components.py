import numpy as np
import cv2 as cv

img_c = cv.imread("test_pictures/multidots_black.jpg")
img = cv.imread("test_pictures/multidots_black.jpg", 0)
height, width = img.shape
print(img.shape)
img = cv.threshold(img, 20, 255, cv.THRESH_BINARY)[1]

output = cv.connectedComponentsWithStats(img, 4, cv.CV_32S)
num_labels = output[0]
labels = output[1]
stats = output[2]
centroids = output[3]

centroids = np.uint16(np.around(centroids))

for i in range(0, num_labels):
	center = (centroids[i][0], centroids[i][1])
	cv.circle(img_c, center, 25, (255,0,0), 2)

cv.namedWindow('test', cv.WINDOW_NORMAL)
cv.imshow('test',img_c)

cv.waitKey(0)
cv.destroyAllWindows()