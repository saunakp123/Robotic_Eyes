import cv2 as cv
import numpy as np
import datetime

SAVE_VIDEO = False
flag = False

cap = cv.VideoCapture(2)
current = datetime.datetime.now()
current = current.strftime("%H-%M-%S")

IS_FOUND = False

while(cap.isOpened()):

	ret, img = cap.read()
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

		rect = cv.minAreaRect(cont)
		(x, y), (width, height), angle = rect
		# print(int(x), int(y), int(width), int(height))

		if (width < height):
			temp = width
			width = height
			height = temp
			angle = angle + 90

		ymin = int(y - height/2)
		xmin = int(x - width/2)
		ymax = int(y + height/2)
		xmax = int(x + width/2)
		ratio = width/height
		print(ratio)

		center = (int(x), int(y))
		M = cv.getRotationMatrix2D(center, angle, 1)
		rot_img = cv.warpAffine(img, M, (row, col))

		cv.drawContours( img, [approx], -1, ( 255, 0, 0 ), 2 )
		# print(ymin, ymax, xmin, xmax)
		roi = rot_img[ymin:ymax, xmin:xmax]
		roi = cv.resize(roi, (int(ratio*col), col), interpolation = cv.INTER_CUBIC)
		IS_FOUND = True


	# cv.imshow('edges', edges)
	cv.imshow('img', img)
	if (IS_FOUND):
		# cv.imshow('rot', rot_img)
		cv.imshow('roi', roi)

	if (not flag and SAVE_VIDEO):
		flag = True
		fourcc_img = cv.VideoWriter_fourcc(*'XVID')
		fourcc_mask = cv.VideoWriter_fourcc(*'XVID')
		path_img = '../test_results/'+current+'_img.avi'
		path_mask = '../test_results/'+current+'_mask.avi'
		out_img = cv.VideoWriter(path_img, fourcc_img, 20.0, (640,480))
		out_mask = cv.VideoWriter(path_mask, fourcc_mask, 20.0, (640,480), 0)

	if flag:
		out_img.write(img)
		out_mask.write(edges)

	key = cv.waitKey(1) & 0xFF
	if key == ord('s'):
		cv.imwrite('test_results/intersection.jpg',img)
	elif key == ord('q'):
		break
	elif key == ord('v'):
		SAVE_VIDEO = True

cap.release()
if (SAVE_VIDEO):
	out_img.release()
	out_mask.release()
cv.destroyAllWindows()
