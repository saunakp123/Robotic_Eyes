import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
# The text to display when intersection is detected
text = "Intersection Detected"
SAVE_VIDEO = False

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
	gray = cv.cvtColor(img_blur,cv.COLOR_BGR2GRAY)
	# fps = cap.get(5)
	height, width, channel = img.shape

	low_range = np.array([150, 100, 220])
	high_range = np.array([250, 200, 255])
	
	img = cv.threshold(img, 20,255, cv.THRESH_BINARY)[1]  # ensure binary
	output = cv.connectedComponentsWithStats(img, 4,cv.CV_32S)
	num_labels = output[0]
	labels = output[1]
	stats = output[2]
# idx = [0,3,1,2,4]
# stats = stats[idx]
	centroid = output[3]
	center = []
	radius = [] 
# print(stats[1][0])
# print(num_labels)
# print(centroid)
	for i in range(1,num_labels):
	
#	center.append(i) = [centroid[i][0],centroid[i][1]]
		center.append([centroid[i][0], centroid[i][1]])
		radius.append([centroid[i][0]-stats[i][0]])
	centerfine = np.uint16(np.around(center))
	radiusfine = np.uint16(np.around(radius))
	print(centerfine)
	print(radiusfine)
	
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
	# circle = concatenate(centrefine,radiusfine)
	# print(circle)
	for i in range(0,num_labels-1):
		cv.circle(img,(centerfine[i][0],centerfine[i][1]),radiusfine[i],(0,255,0),3)
		if (np.linalg.norm(centrefine[0:2] - center) < radiusfine[i] ):
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
# for i in range(0,num_labels):
	# mask = (labels==i).astype(int)
	# points = cv.findNonZero(mask)
	# print(points)
	# avg = np.mean(points, axis=0)
	# avg = avg[0]
	# coord = (avg[0]/height, avg[1]/width)
	# center[i] = (int(avg[0]), int(avg[1]))
	# print(center)
# print(center)
# for i in labels:

# label_hue = np.uint8(179*labels/np.max(labels))
# blank_ch = 255*np.ones_like(label_hue)
# labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

#cvt to BGR for display
# labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

#set bg label to black
# labeled_img[label_hue==0] = 0

#cv.namedWindow('labeled.png', cv.WINDOW_NORMAL)
#cv.imshow('labeled.png', labeled_img)
#cv.imshow('labeledimage.png',labels)
