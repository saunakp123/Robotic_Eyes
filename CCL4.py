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
	

	print(num_labels)

	for i in range(0,num_labels):
		cal_center.append([centroid[i][0], centroid[i][1]])
		cal_radius.append([centroid[i][0] - stats[i][0]])
	cal_center = np.uint16(np.around(cal_center))
	cal_radius = np.uint16(np.around(cal_radius))
	
	# circle = concatenate(centrefine,radiusfine)
	# print(circle)
	for i in range(0,num_labels):
		cv.circle(img, (cal_center[i][0], cal_center[i][1]), cal_radius[i], (0,255,0), 3)
		if (np.linalg.norm(cal_center[0:2] - center) < cal_radius[i] ):
			cv.putText(img, text, (40, 50), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
	
	cv.imshow("capture", img)
	cv.imshow("mask1", mask1)
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