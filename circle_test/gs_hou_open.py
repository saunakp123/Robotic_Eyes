import numpy as np
import cv2 as cv
import datetime

SAVE_VIDEO = False
flag = False

cap = cv.VideoCapture(0)
current = datetime.datetime.now()
current = current.strftime("%H-%M-%S")


while(cap.isOpened()):
	ret, img = cap.read()
	img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	# Preprocess the image with a median blur to make it more robust
	# img_blur = cv.medianBlur(img,5)
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	mb = cv.medianBlur(gray, 5)
	gb = cv.GaussianBlur(gray, (5,5), 0)
	_, th1 = cv.threshold(gb, 127, 255, cv.THRESH_BINARY_INV)
	th2 = cv.adaptiveThreshold(gb, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)
	th3 = cv.adaptiveThreshold(gb, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
	ret3, otsu = cv.threshold(gb, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
	_, fine = cv.threshold(gb, ret3+45, 255, cv.THRESH_BINARY_INV)

	# cv.imshow('gray', gray)
	# cv.imshow('th1', th1)
	# cv.imshow('th2', th2)
	# cv.imshow('th3', th3)
	# cv.imshow('gb', gb)
	# cv.imshow('otsu', otsu)
	# cv.imshow('fine', fine)

	# fps = cap.get(5)
	height, width, channel = img.shape

	low_range = np.array([150, 100, 220])
	high_range = np.array([250, 200, 255])

	# start = time.time()
	# Use Hough method to get calibration circles
	circles1 = cv.HoughCircles(mb,cv.HOUGH_GRADIENT,1,
	100,param1=100,param2=10,minRadius=3,maxRadius=30)
	circles = circles1[0,:,:]
	circles = np.uint16(np.around(circles))

	if (len(circles) != 4):
		print(len(circles))

	for i in circles:
		cv.circle(img,(i[0],i[1]),20,(0,255,0),3)

	cv.imshow("img", img)
	# cv.imshow("mask", mask)
	# Video writer
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
		out_mask.write(mb)

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