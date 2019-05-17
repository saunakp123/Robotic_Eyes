import maestro
import numpy as np
import math as m
import cv2 as cv
import time
#import usefuncs.py 

RATIO = 1.73
COL = 480
ROW = 480*1.73

def go_hor(channel,warp,direction):
    R_hor = channel
    R_hor_num = 0
    h_increment = 60

    M = warp
##    s.setAccel(R_hor,2)
##    s.setSpeed(R_hor,5)
    
    pixel_values = np.zeros((1,2))
    servo_angles = np.zeros((1,2))
    while(True):
        ret, img = cap.read()
        warped_img = getWarp(img, M)
        cv.imshow('image ',warped_img)
        if cv.waitKey(1) & 0xFF == ord('q'):
                break
        laser_pos = get_laser_pos(warped_img)
        #print(laser_pos)
        if (laser_pos):
            servo_angles = np.append(servo_angles,[[s.getPosition(R_hor),6000] ],axis=0)
            pixel_values = np.append(pixel_values,[laser_pos],axis=0)
            R_hor_num = R_hor_num + 1
            if(direction == 'R' or direction == 'r'):
                s.setTarget(R_hor, 6000 - (h_increment*R_hor_num))
            elif(direction == 'L' or direction == 'l'):
                s.setTarget(R_hor, 6000 + (h_increment*R_hor_num))
            time.sleep(1)
             
        else:
            s.setTarget(R_hor, 6000)
            time.sleep(2)
            servo_angles = np.delete(servo_angles,0,axis=0)
            pixel_values = np.delete(pixel_values,0,axis=0)
            break
    return pixel_values, servo_angles


def go_ver(channel,warp,direction):
    R_ver = channel
    R_ver_num = 0
    h_increment = 30
    
##    s.setAccel(R_ver,1)
##    s.setSpeed(R_ver,5)
    M = warp
    
    pixel_values = np.zeros((1,2))
    servo_angles = np.zeros((1,2))
    while(True):
        ret, img = cap.read()
        warped_img = getWarp(img, M)
        cv.imshow('image ',warped_img)
        if cv.waitKey(1) & 0xFF == ord('q'):
                break
        laser_pos = get_laser_pos(warped_img)
        print(laser_pos)
        if (laser_pos):
            servo_angles = np.append(servo_angles,[[6000,s.getPosition(R_ver)]],axis=0)
            pixel_values = np.append(pixel_values,[laser_pos],axis=0)
            R_ver_num = R_ver_num + 1
            if(direction == 'd' or direction == 'D'):
                s.setTarget(R_ver, 6000 + (h_increment*R_ver_num))
            elif(direction == 'u' or direction == 'U'):
                s.setTarget(R_ver, 6000 - (h_increment*R_ver_num))
            time.sleep(1)
             
        else:
            s.setTarget(R_ver, 6000)
            time.sleep(2)
            servo_angles = np.delete(servo_angles,0,axis=0)
            pixel_values = np.delete(pixel_values,0,axis=0)
            break
    return pixel_values, servo_angles


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

def getWarpMatrix(img):
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
		print("COL: ", COL)
		ROW = col*RATIO

		des = np.float32([[0,0],[ROW,0],[ROW,COL],[0,COL]])

		M = cv.getPerspectiveTransform(box, des)

	if (IS_FOUND):
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
	cv.imshow('mask ',mask)
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
def getAngle(x,coefficient):
    angle = np.polyval(coefficient,x)
    return angle

def goToPos(angle,channel, pausetime = 1):
    time.sleep(pausetime)
    s.setTarget(channel,int(angle))
    time.sleep(pausetime)
    
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
	print(circles1)
	circles = circles1[0,:,:]
	circles = np.uint16(np.around(circles))
	print(circles)	
	print('Total number of circles found: ', len(circles))

	img_copy = img.copy()
	for i in circles:
		cv.circle(img_copy,(i[0],i[1]),20,(0,255,0),3)

	return circles

	cv.imshow("circle", img_copy)
	# cv.imshow("mask", mask)    


def P_control_X(laser_pos, desired_pos, channel, Kp):
    
    x_difference = laser_pos[0] - desired_pos[0]
    
    x_compensation = round(Kp*x_difference)
    #print(x_compensation)
    prev_servo_pos = s.getPosition(channel)
    print("prev position x: ", prev_servo_pos)
    if (laser_pos[0] > desired_pos[0]):
        x_final = prev_servo_pos + (x_compensation)
    else:
        x_final = prev_servo_pos - x_compensation
    print(x_final)
    goToPos(x_final, channel, 2)
    time.sleep(1)
    return

def P_control_Y(laser_pos, desired_pos, channel, Kp):
    y_difference = laser_pos[1] - desired_pos[1]
    y_compensation = round(Kp*y_difference)
    #print(y_compensation)
    prev_servo_pos = s.getPosition(channel)
    print("prev position y: ",prev_servo_pos)
    if (laser_pos[1] > desired_pos[1]):
        y_final = prev_servo_pos - (y_compensation)
    else:
        y_final = prev_servo_pos + (y_compensation)
    print(y_final)
    goToPos(y_final, channel, 2)
    time.sleep(1)
    return

def P_mega_control(desired_pos, channel_x, channel_y, warp, Kp):
    s.setAccel(R_hor,1)
    s.setSpeed(R_hor,1)
    tolerance = 12
    tol = tolerance
    M = warp
    while(True):
        ret, img = cap.read()
        warped_img = getWarp(img, M)
        laser_pos = get_laser_pos(warped_img)
        print("mega control laser pos: ",laser_pos)
        if(abs(laser_pos[0]-desired_pos[0])<tol and abs(laser_pos[1]-desired_pos[1])<tol):
            break
        if(abs(laser_pos[0]-desired_pos[0])>tol):
            P_control_X(laser_pos, desired_pos, channel_x, Kp)
        if(abs(laser_pos[1]-desired_pos[1])>tol):
            P_control_Y(laser_pos, desired_pos, channel_y, Kp)
        ret, img = cap.read()
        warped_img = getWarp(img, M)
        laser_pos = get_laser_pos(warped_img)
        x_error = abs(laser_pos[0] - desired_pos[0])
        y_error = abs(laser_pos[1] - desired_pos[1])
        if x_error < tolerance and y_error < tolerance:
            break
    return

def convergence(desired_pos,laser_pos,tol,R_hor,R_ver,inc):
    desx = desired_pos[0]
    desy = desired_pos[1]
    while(abs(laser_pos[0]-desx) > tol or abs(laser_pos[1]-desy) > tol):
        cur_posX = s.getPosition(R_hor)
        cur_posY = s.getPosition(R_ver)
        print("cur_pos: ", [cur_posX, cur_posY])
        if (laser_pos[0]>desx):
            print(1)
            new_posX = s.setTarget(R_hor,cur_posX+inc)
            time.sleep(1)
        elif(laser_pos[0]<desx):
            print(2)
            new_posX = s.setTarget(R_hor,cur_posX-inc)
            time.sleep(1)
        if (laser_pos[1]>desy):
            print(3)
            new_posX = s.setTarget(R_ver,cur_posY-inc)
            time.sleep(1)
        elif(laser_pos[1]<desy):
            print(4)
            new_posX = s.setTarget(R_ver,cur_posY+inc)
            time.sleep(1)
        ret, img = cap.read()
        warped_img = getWarp(img, M)
        laser_pos = get_laser_pos(warped_img)
        time.sleep(1)
    print("new pos: ",laser_pos)
    return
    


##############################################################################

s = maestro.Controller('COM6')

# assigning channels to the servos
'''R_ver = 
L_hor =
L_ver ='''
R_ver = 7
R_hor = 8

cap = cv.VideoCapture(0)
ret, img = cap.read()
M = getWarpMatrix(img)
input("press any key to continue: ")
s.setAccel(R_hor,0)
s.setSpeed(R_hor,0)
s.setTarget(R_hor,6000)
s.setTarget(R_ver,6000)
time.sleep(2)
# Getting horizontal pixels 
pixel_values_r, servo_angles_r = go_hor(R_hor,M,'r')
pixel_values_l, servo_angles_l = go_hor(R_hor,M,'l')
pixel_values_l = np.delete(pixel_values_l,0,axis=0)
pixel_values_l = np.flip(pixel_values_l,axis=0)
servo_angles_l =  np.delete(servo_angles_l,0,axis=0)
servo_angles_l = np.flip(servo_angles_l,axis=0)
#Making pixels and servo angle list for horizontal
pixel_values_hor = np.append(pixel_values_l,pixel_values_r,axis = 0)
servo_angles_hor = np.append(servo_angles_l,servo_angles_r,axis = 0)
#Getting coefficients for horizontal 
x_pix_data = [i[0] for i in pixel_values_hor]
x_angle_data = [i[0] for i in servo_angles_hor]
x_coeffs = np.polyfit(x_pix_data, x_angle_data, 5)

# Getting vertical pixels 
pixel_values_up, servo_angles_up = go_ver(R_ver,M,'u')
pixel_values_down, servo_angles_down = go_ver(R_ver,M,'d')
pixel_values_up = np.delete(pixel_values_up,0,axis=0)
pixel_values_up = np.flip(pixel_values_up,axis=0)
servo_angles_up =  np.delete(servo_angles_up,0,axis=0)
servo_angles_up = np.flip(servo_angles_up,axis=0)
#Making pixels and servo angle list for horizontal
pixel_values_ver = np.append(pixel_values_up,pixel_values_down,axis = 0)
servo_angles_ver = np.append(servo_angles_up,servo_angles_down,axis = 0)
#Getting coefficients for horizontal 
y_pix_data = [i[1] for i in pixel_values_ver]
y_angle_data = [i[1] for i in servo_angles_ver]
y_coeffs = np.polyfit(y_pix_data, y_angle_data, 5)

x_dest = [700, 200, 200, 700, 700] 
y_dest = [100, 100, 350, 350, 100]

x_theta = [0]*5 ;
y_theta = [0]*5 ;

for i in range(5):
    
    dest = (x_dest[i], y_dest[i])

    x_theta[i] = getAngle(x_dest[i],x_coeffs)
    goToPos(x_theta[i],R_hor,0)

    y_theta[i] = getAngle(y_dest[i],y_coeffs)
    goToPos(y_theta[i],R_ver,0)

    time.sleep(1) 

    ret, img = cap.read()
    warped_img = getWarp(img, M)
    cv.imshow('image ',warped_img)
    laser_pos = get_laser_pos(warped_img)
    print(laser_pos)
    convergence(dest,laser_pos,10,R_hor,R_ver,10)
cap.release()
