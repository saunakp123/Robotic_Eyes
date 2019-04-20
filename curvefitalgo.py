import maestro
import numpy as np
import math as m
import cv2 as cv
import time

RATIO = 1.7
COL = 0
ROW = 0

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def getWarpMatrix(img, ratio):
	global COL, ROW
	IS_FOUND = False
	col, row = img.shape[:2]
	# Preprocess the image with a median blur to make it more robust
	# img_blur = cv.medianBlur(img,5)
	gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
	gray = cv.bilateralFilter( gray, 1, 10, 120 )

	edges  = cv.Canny( gray, 10, 250 )
	kernel = cv.getStructuringElement( cv.MORPH_RECT, ( 7, 7 ) )
	closed = cv.morphologyEx( edges, cv.MORPH_CLOSE, kernel )

	contours, h = cv.findContours( closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )

	for cont in contours:
		if cv.contourArea( cont ) < 20000 : continue

		# print(cv.contourArea( cont ))
		arc_len = cv.arcLength( cont, True )
		approx = cv.approxPolyDP( cont, 0.1 * arc_len, True )

		if ( len( approx ) != 4 ): continue
		IS_FOUND = True

		box = np.array([
			approx[0][0],
			approx[1][0],
			approx[2][0],
			approx[3][0]], dtype = "float32")

		box = order_points(box)

		COL = col
		ROW = col*ratio

		des = np.float32([[0,0],[ROW,0],[ROW,COL],[0,COL]])
		
		M = cv.getPerspectiveTransform(box, des)

	if (IS_FOUND):
		print("screen found")
		return M
	else:
		print("screen not found")
		return 0

	
def getWarp(img, M):
	dst = cv.warpPerspective(img,M,(int(ROW),COL))

	return dst


def get_laser_pos(img):
	height, width, channel = img.shape

	low_range = np.array([150, 100, 220])
	high_range = np.array([250, 200, 255])

	mask = cv.inRange(img, low_range, high_range)
	points = cv.findNonZero(mask)

	if (points is None):
		return 0
	else:
		# Average these points
		avg = np.mean(points, axis=0)
		avg = avg[0]
		coord = (avg[0]/height, avg[1]/width)

		center = (int(avg[0]), int(avg[1]))

	cv.circle(img, center, 5, (0,255,0), -1)

	return center


s = maestro.Controller('COM4')

# assigning channels to the servos
R_hor = 5
'''R_ver = 
L_hor =
L_ver ='''

# setting all servos to home
s.setTarget(R_hor, 6000)
#s.setTarget(R_ver, 6000)
##s.setTarget(L_hor, 6000)
##s.setTarget(L_ver, 6000)

# Initialization
increment = 80
R_hor_Right_limit = 0
R_hor_Left_limit = 0
velocity = s.setSpeed(R_hor,1)
h_points = 6
# Capturing image & getting warped image parameter
cap = cv.VideoCapture(0)
ret, img = cap.read()
M = getWarpMatrix(img, RATIO)
input("press key to turn screen black: ");
cap.release()
cap = cv.VideoCapture(0)
# Loop to obtain screen limits
while(True):
    ret, img = cap.read()
    cv.imshow('img', img)
    warped_img = getWarp(img, M)
    cv.imshow('warp', warped_img)
    if cv.waitKey(1) & 0xFF == ord('q'):
	    break
    laser_pos = get_laser_pos(warped_img)
    print(laser_pos)
# Get left side limit
    for i in range(1000)    
        if (laser_pos):
            s.setTarget(R_hor, 6000 + (i*increment))
            time.sleep(3)  
        else:
            R_hor_Left_limit = 6000 + i*increment
            s.setTarget(R_hor, 6000)
            break
# Get right side limit
    for i in range(1000)    
        if (laser_pos):
            s.setTarget(R_hor, 6000 - (i*increment))
            time.sleep(3)  
        else:
            R_hor_Right_limit = 6000 - i*increment
            s.setTarget(R_hor, 6000)
            break
# Getting the horizontal sweep and finding the increment angles
horizontal_sweep = R_hor_Left_limit - R_hor_Right_limit
h_increment = horizontal_sweep/(h_points - 1)
# Initializing pixel array 
horizontal_pixel_array = np.zeros(h_points)
# Move and Store left limit position in horizontal pixel array
s.setTarget(R_hor, R_hor_Left_limit)
time.sleep(5)
horizontal_pixel_array[0],- = get_laser_pos(warped_img)
horizontal_angle_array[0] = R_hor_Left_limit
# Loop to store pixel positions and angles in respective arrays
for i in range(1,h_points):
    s.setTarget(R_hor, R_hor_Left_limit - i*h_increment)
    time.sleep(3)
    horizontal_angle_array[i] = R_hor_Left_limit - i*h_increment
    horizontal_pixel_array[i] = get_laser_pos(warped_img)
# Normalizing horizontal pixel array by a factor of 100
scaling_factor = 100
norm_horizontal_pixel array = horizontal_pixel_array/scaling_factor
# Curve fitting pixels and angles using polyfit
coefficient = polyfit(norm_horizontal_pixel_array, horizontal_angle_array, h_points-1)
# Function to return angle on pixel input
def getAngle(x,coefficient,scaling_factor):
    x = x/scaling_factor
    theta = 0
    for i in range(h_points):
        theta = theta + (x**i)*coefficient[i]
    return theta
# Function to move servo to given pixel position
def goToPos(x):
    angle = getAngle(x,coefficient)
    s.setTarget(R_hor,angle)
