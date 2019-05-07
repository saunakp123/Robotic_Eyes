def getCircle(img):
	# img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV)
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

	print('Total number of circles found: ', len(circles))

	img_copy = img.copy()
	for i in circles:
		cv.circle(img_copy,(i[0],i[1]),20,(0,255,0),3)

	cv.imshow("circle", img_copy)
	# cv.imshow("mask", mask)

	return circles