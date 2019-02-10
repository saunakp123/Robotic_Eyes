import numpy as np
import cv2 as cv
import datetime

# Change this to True if you want to make a video
SAVE_VIDEO = False

cap = cv.VideoCapture(0)
# The text to display when intersection is detected
text = "Intersection Detected"
save_video = False

# Video writer
if (SAVE_VIDEO):
	fourcc = cv.VideoWriter_fourcc(*'XVID')
	out = cv.VideoWriter('test_results/output.avi', fourcc, 20.0, (640,480))

while(cap.isOpened()):
	ret, img = cap.read()
	# Preprocess the image with a median blur to make it more robust
	img_blur = cv.medianBlur(img,5)
	gray = cv.cvtColor(img_blur,cv.COLOR_BGR2GRAY)
	# fps = cap.get(5)
	height, width, channel = img.shape

	low_range = np.array([150, 100, 220])
	high_range = np.array([250, 200, 255])

	# start = time.time()
	# Use Hough method to get calibration circles
	circles1 = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,
	100,param1=100,param2=10,minRadius=3,maxRadius=30)
	circles = circles1[0,:,:]
	circles = np.uint16(np.around(circles))

	if (len(circles) != 4):
		print(len(circles))

	mask = cv.inRange(img, low_range, high_range)
	points = cv.findNonZero(mask)
	if (points is None):
		center = ((0, 0))
	else:
		# Average these points
		avg = np.mean(points, axis=0)
		avg = avg[0]
		coord = (avg[0]/height, avg[1]/width)

		center = (int(avg[0]), int(avg[1]))

	# end = time.time()
	# print(end - start)

	# Draw circle aroung the laser dot
	# cv.circle(img, center, 5, (0,255,0), -1)

	# Draw something if intersection is detected
	# Can be simplified if camera is not moving (calibration points stay the same)
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