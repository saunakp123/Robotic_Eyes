def getAngle(x,coefficient):
    angle = np.polyval(coefficient,x)
    return angle

def goToPos(angle,channel):
    time.sleep(3)
    s.setTarget(channel,int(angle))
    time.sleep(5)
    
def getCircle(img):
	img_hue = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	img_blur = cv.medianBlur(img,5)
	# img_blur = cv.GaussianBlur(img,(5,5),0)

	# fps = cap.get(5)
	height, width, channel = img.shape

	# The HSV range
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
	labeled_img[labels == 0] = 0
	cv.imshow('labeled_img', labeled_img)
	
	circles1 = cv.HoughCircles(labeled_img,cv.HOUGH_GRADIENT,1,
	100,param1=100,param2=2,minRadius=2,maxRadius=30)
	circles = circles1[0,:,:]
	circles = np.uint16(np.around(circles))
		
	print('Total number of circles found: ', len(circles))

	img_copy = img.copy()
	for i in circles:
		cv.circle(img_copy,(i[0],i[1]),20,(0,255,0),3)

	return circles

	cv.imshow("circle", img_copy)
	# cv.imshow("mask", mask)    
