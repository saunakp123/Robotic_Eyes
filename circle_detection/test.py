import cv2 as cv
import numpy as np
import datetime

from transform import order_points
from transform import four_point_transform
SAVE_VIDEO = False
flag = False

cap = cv.VideoCapture(0)
current = datetime.datetime.now()
current = current.strftime("%H-%M-%S")

RATIO = 1.73

IS_FOUND = False


while(cap.isOpened()):

	ret, img = cap.read()
	col, row = img.shape[:2]
	col2 = col
	row2 = int(col*RATIO)
	dst = img
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

		des = np.float32([[0,0],[row2,0],[row2,col2],[0,col2]])

		M = cv.getPerspectiveTransform(box, des)

		dst = cv.warpPerspective(img,M,(int(row2),col2))

		# cv.drawContours( img, [approx], -1, ( 255, 0, 0 ), 2 )
		# for i in range(4):
		# 	cv.putText(img, str(i), (box[i][0], box[i][1]), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

	# if (IS_FOUND == False): continue
	dst_hue = cv.cvtColor(dst, cv.COLOR_BGR2HSV)
	low_range2 = np.array([100, 100, 100])
	high_range2 = np.array([120, 255, 255])
	mask = cv.inRange(dst_hue, low_range2, high_range2)
	output = cv.connectedComponentsWithStats(mask, 8,cv.CV_32S)
	num_labels = output[0]
	labels = output[1]
	stats = output[2]
	centroid = output[3]
	cal_center = []
	cal_radius = []
	
	# print(num_labels)
	for i in range(0,num_labels):
		radius = centroid[i][0] - stats[i][0]
		if radius > 8 or radius < 3: continue
		cal_center.append([centroid[i][0], centroid[i][1]])
		cal_radius.append([radius])
	cal_center = np.uint16(np.around(cal_center))
	cal_radius = np.uint16(np.around(cal_radius))
	
	num = cal_radius.size
	for i in range(0,num):
		cv.circle(dst, (cal_center[i][0], cal_center[i][1]), cal_radius[i], (0,255,0), 3)

	cv.imshow('img', img)
	if (IS_FOUND):
		cv.imshow('dst', dst)

	if (not flag and SAVE_VIDEO):
		flag = True
		fourcc_img = cv.VideoWriter_fourcc(*'XVID')
		fourcc_mask = cv.VideoWriter_fourcc(*'XVID')
		path_img = '../test_results/'+current+'_img.avi'
		path_mask = '../test_results/'+current+'_mask.avi'
		out_img = cv.VideoWriter(path_img, fourcc_img, 20.0, (row,col))
		out_mask = cv.VideoWriter(path_mask, fourcc_mask, 20.0, (row2,col2))

	if flag:
		out_img.write(img)
		out_mask.write(dst)

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
