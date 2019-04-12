import cv2 as cv
import numpy as np

def getWarp(img):
	IS_FOUND = False
	col, row = img.shape[:2]
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

		rect = cv.minAreaRect(cont)
		(x, y), (width, height), angle = rect
		box = cv.boxPoints(rect)

		if (width < height):
			swap = np.zeros((4, 2))
			for i in range(0, 4):
				swap[i] = box[(i+1)%4]
			box = swap

		col2 = col
		row2 = col*RATIO

		pts1 = np.float32(box)
		pts2 = np.float32([[0,col2],[0,0],[row2,0],[row2,row2]])
		# print(pts1)
		# pts2 = np.float32([[0,300],[0,0],[300,0],[300,300]])
		M = cv.getPerspectiveTransform(pts1,pts2)

		dst = cv.warpPerspective(img,M,(int(row2),col2))
		# dst = cv.warpPerspective(img,M,(300,300))

		cv.drawContours( img, [approx], -1, ( 255, 0, 0 ), 2 )

	if (IS_FOUND):
		return dst
	else:
		print("screen not found")
		return img