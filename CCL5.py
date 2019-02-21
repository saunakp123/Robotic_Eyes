import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
# The text to display when intersection is detected
text = "Intersection Detected"
SAVE_VIDEO = False

save_video = False

# Video writer
if (SAVE_VIDEO):
	fourcc = cv.VideoWriter_fourcc(*'XVID')
	out = cv.VideoWriter('test_results/output.avi', fourcc, 20.0, (640,480))
while(cap.isOpened()):
	# img = cv.imread('D:/Personal/UW/Robotic Eyes/multidots_black.jpg',0)
	# height,width = img.shape
	ret, img = cap.read()
	# Preprocess the image with a median blur to make it more robust
	img_blur = cv.medianBlur(img,5)
	# fps = cap.get(5)
	height, width, channel = img.shape

	low_range1 = np.array([150, 100, 220])
	high_range1 = np.array([250, 200, 255])

	# Find the laser dot
	mask1 = cv.inRange(img, low_range1, high_range1)
	points = cv.findNonZero(mask1)
	if (points is None):
		center = ((0, 0))
	else:
		# Average these points
		avg = np.mean(points, axis=0)
		avg = avg[0]
		coord = (avg[0]/height, avg[1]/width)

		center = (int(avg[0]), int(avg[1]))

	
	# Blur or no blur?
	low_range2 = np.array([200, 10, 10])
	high_range2 = np.array([255, 100, 100])
	#gray = cv.cvtColor(img_blur,cv.COLOR_BGR2GRAY)
	#gray = cv.threshold(gray, 20,255, cv.THRESH_BINARY)[1]  # ensure binary
	mask2 = cv.inRange(img, low_range2, high_range2)
	output = cv.connectedComponentsWithStats(mask2, 4,cv.CV_32S)
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
	labeled_img[label_hue==0] = 0
	
	circles1 = cv.HoughCircles(labeled_img,cv.HOUGH_GRADIENT,1,
	100,param1=100,param2=10,minRadius=3,maxRadius=30)
	circles = circles1[0,:,:]
	circles = np.uint16(np.around(circles))

	if (len(circles) != 4):
		print(len(circles))

	for i in circles:
		cv.circle(img,(i[0],i[1]),20,(0,255,0),3)
		if (np.linalg.norm(i[0:2] - center) < i[2] ):
			cv.putText(img, text, (40, 50), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

	cv.imshow("capture", img)
	cv.imshow("mask", mask)

	if save_video:
		out.write(img)

	key = cv.waitKey(1) & 0xFF
	if key == ord('s'):
		cv.imwrite('test_results/intersection.jpg',img)
	elif key == ord('q'):
		break
	elif key == ord('v') and SAVE_VIDEO:
		save_video = True

cap.release()
out.release()
cv.destroyAllWindows()
