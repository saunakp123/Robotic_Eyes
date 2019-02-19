import cv2 as cv
import numpy as np
# Change this to True if you want to make a video
SAVE_VIDEO = True

cap = cv.VideoCapture(0)
# The text to display when intersection is detected
text = "Intersection Detected"
save_video = False

# Video writer
if (SAVE_VIDEO):
	fourcc = cv.VideoWriter_fourcc(*'XVID')
	out = cv.VideoWriter('D:/Personal/UW/Robotic Eyes/output.avi', fourcc, 20.0, (640,480))
while(cap.isOpened()):

	ret, img = cap.read()
	# Preprocess the image with a median blur to make it more robust
	img_blur = cv.medianBlur(img,5)
	
	height, width, channel = img.shape

	low_range1 = np.array([10, 10, 220])
	high_range1 = np.array([100, 100, 255])

	# Find the laser dot
	mask1 = cv.inRange(img_blur, low_range1, high_range1)
	points = cv.findNonZero(mask1)
	if (points is None):
		center = ((0, 0))
	else:
		# Average these points
		avg = np.mean(points, axis=0)
		avg = avg[0]
		coord = (avg[0]/height, avg[1]/width)

		center = (int(avg[0]), int(avg[1]))

	
	# Blur 
	low_range2 = np.array([220, 10, 10])
	high_range2 = np.array([255, 100, 100])
	mask2 = cv.inRange(img_blur, low_range2, high_range2)
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
		cv.circle(img_blur, (cal_center[i][0], cal_center[i][1]), cal_radius[i], (0,255,0), 3)
		if (np.linalg.norm(cal_center[0:2] - center) < cal_radius[i] ):
			cv.putText(img_blur, text, (40, 50), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
	
	cv.imshow("capture", img_blur)
	cv.imshow("mask1", mask1)
	if save_video:
		out.write(img_blur)

	key = cv.waitKey(1) & 0xFF
	if key == ord('s'):
		cv.imwrite('test_results/intersection.jpg',img_blur)
	elif key == ord('q'):
		break
	elif key == ord('v') and SAVE_VIDEO:
		save_video = True

cap.release()
out.release()
cv.destroyAllWindows()
