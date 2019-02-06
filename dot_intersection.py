import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
text = "Intersection Detected"
while(1):
	# get a frame
	ret, img = cap.read()
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	# get the fps of the camera
	fps = cap.get(5)
	# opencv index use height*width
	height, width, channel = img.shape

	low_range = np.array([150, 100, 220])
	high_range = np.array([250, 200, 255])

	# start = time.time()
	# use Hough method to get calibration circles
	circles1 = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,
	100,param1=100,param2=30,minRadius=10,maxRadius=100)
	circles = circles1[0,:,:]
	circles = np.uint16(np.around(circles))

	if (len(circles) != 4):
		print(len(circles))

	mask = cv.inRange(img, low_range, high_range)
	points = cv.findNonZero(mask)
	if (points is None):
		center = ((0, 0))
	else:	
		# average these points
		avg = np.mean(points, axis=0)
		avg = avg[0]
		coord = (avg[0]/height, avg[1]/width)

		center = (int(avg[0]), int(avg[1]))

	# end = time.time()
	# print(end - start)

	# cv.circle(img, center, 5, (0,255,0), -1)

	# draw something if intersection is detected
	# can be simplified if camera is not moving (calibration points stay the same)
	for i in circles:
		if (np.linalg.norm(i[0:2] - center) < i[2] ):
			cv.putText(img, text, (40, 50), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
		# cv.circle(img,(i[0],i[1]),20,(0,255,0),3)

	# show a frame
	cv.imshow("capture", img)
	cv.imshow("mask", mask)
	# quit if 'q' is pressed
	if cv.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv.destroyAllWindows()