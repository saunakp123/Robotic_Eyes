import cv2 as cv
import numpy as np
import datetime

SAVE_VIDEO = False
flag = False

cap = cv.VideoCapture(1)
current = datetime.datetime.now()
current = current.strftime("%H-%M-%S")

while(cap.isOpened()):

	ret, img = cap.read()
	# Preprocess the image with a median blur to make it more robust
	# img_blur = cv.medianBlur(img,5)
	img_blur = cv.GaussianBlur(img,(5,5),0)
	img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	height, width, channel = img.shape
	
	low_range2 = np.array([100, 100, 100])
	high_range2 = np.array([120, 255, 255])
	mask2 = cv.inRange(img_hue, low_range2, high_range2)
	output = cv.connectedComponentsWithStats(mask2, 8,cv.CV_32S)
	num_labels = output[0]
	labels = output[1]
	stats = output[2]
	centroid = output[3]
	cal_center = []
	cal_radius = []
	

	print(num_labels)

	for i in range(0,num_labels):
		cal_center.append([centroid[i][0], centroid[i][1]])
		cal_radius.append([centroid[i][0] - stats[i][0]])
	cal_center = np.uint16(np.around(cal_center))
	cal_radius = np.uint16(np.around(cal_radius))
	

	for i in range(1,num_labels):
		cv.circle(img, (cal_center[i][0], cal_center[i][1]), cal_radius[i], (0,255,0), 3)
	

	cv.imshow("capture", img)
	# cv.imshow("mask1", mask1)
	cv.imshow("mask2", mask2)

	if (not flag and SAVE_VIDEO):
		flag = True
		fourcc_img = cv.VideoWriter_fourcc(*'XVID')
		fourcc_mask = cv.VideoWriter_fourcc(*'XVID')
		path_img = 'test_results/'+current+'_img.avi'
		path_mask = 'test_results/'+current+'_mask.avi'
		out_img = cv.VideoWriter(path_img, fourcc_img, 20.0, (640,480))
		out_mask = cv.VideoWriter(path_mask, fourcc_mask, 20.0, (640,480), 0)

	if flag:
		out_img.write(img)
		out_mask.write(mask2)

	key = cv.waitKey(1) & 0xFF
	if key == ord('s'):
		cv.imwrite('test_results/intersection.jpg',img)
	elif key == ord('q'):
		break
	elif key == ord('v'):
		SAVE_VIDEO = True

cap.release()
out_img.release()
out_mask.release()
cv.destroyAllWindows()
