import maestro
import numpy as np
import math as m
import cv2 as cv
import time

RATIO = 1.73
COL = 480
ROW = 480*1.73

def go_hor(channel,warp,direction):
    R_hor = channel
    s.setTarget(R_hor, 6000)
    
    time.sleep(3)
    
    R_hor_num = 0
    h_increment = 60

    M = warp
    s.setAccel(R_hor,2)
    s.setSpeed(R_hor,5)
    
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
            if(direction == 'L' or direction == 'l'):
                s.setTarget(R_hor, 6000 - (h_increment*R_hor_num))
            elif(direction == 'R' or direction == 'r'):
                s.setTarget(R_hor, 6000 + (h_increment*R_hor_num))
            time.sleep(1)
             
        else:
            s.setTarget(R_hor, 6000)
            time.sleep(3)
            servo_angles = np.delete(servo_angles,0,axis=0)
            pixel_values = np.delete(pixel_values,0,axis=0)
            break
    return pixel_values, servo_angles


def go_ver(channel,warp,direction):
    R_ver = channel
    s.setTarget(R_ver, 6000)

    R_ver_num = 0
    h_increment = 60
    
    s.setAccel(R_ver,1)
    s.setSpeed(R_ver,5)
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
            if(direction == 'u' or direction == 'U'):
                s.setTarget(R_ver, 6000 + (h_increment*R_ver_num))
            elif(direction == 'd' or direction == 'D'):
                s.setTarget(R_ver, 6000 - (h_increment*R_ver_num))
            time.sleep(2)
             
        else:
            s.setTarget(R_ver, 6000)
            time.sleep(4)
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

def goToPos(angle,channel):
    time.sleep(3)
    s.setTarget(channel,int(angle))
    time.sleep(5)
    
##############################################################################


s = maestro.Controller('COM6')

# assigning channels to the servos
'''R_ver = 
L_hor =
L_ver ='''
R_ver = 7
R_hor = 5

cap = cv.VideoCapture(0)
ret, img = cap.read()
M = getWarpMatrix(img)
input("press any key to continue: ")

pixel_values_r, servo_angles_r = go_hor(R_hor,M,'r')
#print(pixel_values_r)
#print(servo_angles_r)
pixel_values_l, servo_angles_l = go_hor(R_hor,M,'l')
#print(pixel_values_l)
#print(servo_angles_l)

pixel_values_l = np.delete(pixel_values_l,0,axis=0)
pixel_values_l = np.flip(pixel_values_l,axis=0)

servo_angles_l =  np.delete(servo_angles_l,0,axis=0)
servo_angles_l = np.flip(servo_angles_l,axis=0)

pixel_values_hor = np.append(pixel_values_l,pixel_values_r,axis = 0)

servo_angles_hor = np.append(servo_angles_l,servo_angles_r,axis = 0)

x_pix_data = [i[0] for i in pixel_values_hor]
x_angle_data = [i[0] for i in servo_angles_hor]

'''size_pix = np.shape(x_pix_data)
if(size_pix[0]%2!=0):
        x_pix_data = np.delete(x_pix_data,-1)
        x_angle_data = np.delete(x_angle_data,-1)
size_pix = np.shape(x_pix_data)'''
coeffs = np.polyfit(x_pix_data, x_angle_data, 5)
#print(coeffs)
#print(x_pix_data)
#print(x_angle_data)
x_dest = 
theta = getAngle(x_dest,coeffs)
goToPos(theta,R_hor)
ret, img = cap.read()
warped_img = getWarp(img, M)
cv.imshow('image ',warped_img)
laser_pos = get_laser_pos(warped_img)
print(laser_pos[0])
