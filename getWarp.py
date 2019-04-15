import cv2 as cv
import numpy as np

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def getWarp(img):
	IS_FOUND = False
	col, row = img.shape[:2]
	# Preprocess the image with a median blur to make it more robust
	# img_blur = cv.medianBlur(img,5)
	gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
	gray = cv.bilateralFilter( gray, 1, 10, 120 )

	edges  = cv.Canny( gray, 10, 250 )
	kernel = cv.getStructuringElement( cv.MORPH_RECT, ( 7, 7 ) )
	closed = cv.morphologyEx( edges, cv.MORPH_CLOSE, kernel )

	contours, h = cv.findContours( closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )

	for cont in contours:
		if cv.contourArea( cont ) < 20000 : continue

		# print(cv.contourArea( cont ))
		arc_len = cv.arcLength( cont, True )
		approx = cv.approxPolyDP( cont, 0.1 * arc_len, True )

		if ( len( approx ) != 4 ): continue
		IS_FOUND = True

		box = np.array([
			approx[0][0],
			approx[1][0],
			approx[2][0],
			approx[3][0]], dtype = "float32")

		box = order_points(box)

		col2 = col
		row2 = col*RATIO

		des = np.float32([[0,0],[row2,0],[row2,col2],[0,col2]])
		
		M = cv.getPerspectiveTransform(box, des)

		# dst = cv.warpPerspective(img,M,(int(row2),col2))


	if (IS_FOUND):
		return M
	else:
		print("screen not found")
		return 0