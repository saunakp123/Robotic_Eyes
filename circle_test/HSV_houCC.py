import cv2 as cv
import numpy as np
import datetime

SAVE_VIDEO = False
flag = False

cap = cv.VideoCapture(1)
current = datetime.datetime.now()
current = current.strftime("%H-%M-%S")

while(cap.isOpened()):
	# img = cv.imread('D:/Personal/UW/Robotic Eyes/multidots_black.jpg',0)
	# height,width = img.shape
	ret, img = cap.read()
	# Preprocess the image with a median blur to make it more robust
	img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	img_blur = cv.medianBlur(img,5)
	# img_blur = cv.GaussianBlur(img,(5,5),0)

	# fps = cap.get(5)
	height, width, channel = img.shape

	low_range2 = np.array([100, 100, 100])
	high_range2 = np.array([120, 255, 255])
	mask2 = cv.inRange(img_hue, low_range2, high_range2)
	output = cv.connectedComponentsWithStats(mask2, 4, cv.CV_32S)
	num_labels = output[0]
	labels = output[1]
	stats = output[2]
	centroid = output[3]
	cal_center = []
	cal_radius = []
	
	# Map component labels to hue val
	label_hue = np.uint8(179*labels/np.max(labels))
	blank_ch = 255*np.ones_like(label_hue)
	labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

	# cvt to BGR for display
	labeled_img = cv.cvtColor(labeled_img, cv.COLOR_BGR2GRAY)

	# set bg label to black
	labeled_img[labels==0] = 0
	cv.imshow('labeled_img', labeled_img)
	
	circles1 = cv.HoughCircles(labeled_img,cv.HOUGH_GRADIENT,1,
	100,param1=100,param2=2,minRadius=2,maxRadius=30)
	circles = circles1[0,:,:]
	circles = np.uint16(np.around(circles))

	if (len(circles) != 4):
		print(len(circles))

	for i in circles:
		cv.circle(img,(i[0],i[1]),20,(0,255,0),3)

	cv.imshow("capture", img)
	# cv.imshow("mask", mask)

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