import cv2 as cv
import numpy as np
import datetime

SAVE_VIDEO = False
flag = False

cap = cv.VideoCapture(1)
current = datetime.datetime.now()
current = current.strftime("%H-%M-%S")

IS_FOUND = False

_width  = 600.0
_height = 420.0
_margin = 0.0

corners = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)

pts_dst = np.array( corners, np.float32 )

while(cap.isOpened()):

	ret, img = cap.read()
	# Preprocess the image with a median blur to make it more robust
	# img_blur = cv.medianBlur(img,5)
	gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
	gray = cv.bilateralFilter( gray, 1, 10, 120 )

	edges  = cv.Canny( gray, 10, 250 )
	kernel = cv.getStructuringElement( cv.MORPH_RECT, ( 7, 7 ) )
	closed = cv.morphologyEx( edges, cv.MORPH_CLOSE, kernel )

	contours, h = cv.findContours( closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )

	for cont in contours:
		if cv.contourArea( cont ) > 5000 :
			arc_len = cv.arcLength( cont, True )
			approx = cv.approxPolyDP( cont, 0.1 * arc_len, True )

			if ( len( approx ) == 4 ):
				IS_FOUND = True
				#M = cv.moments( cont )
				#cX = int(M["m10"] / M["m00"])
				#cY = int(M["m01"] / M["m00"])
				#cv.putText(img, "Center", (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

				pts_src = np.array( approx, np.float32 )

				h, status = cv.findHomography( pts_src, pts_dst )
				out = cv.warpPerspective( img, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )

				cv.drawContours( img, [approx], -1, ( 255, 0, 0 ), 2 )

			else : pass

	#cv.imshow( 'closed', closed )
	#cv.imshow( 'gray', gray )
	cv.imshow( 'edges', edges )
	cv.imshow( 'img', img )

	if IS_FOUND :
		cv.imshow( 'out', out )

	if (not flag and SAVE_VIDEO):
		flag = True
		fourcc_img = cv.VideoWriter_fourcc(*'XVID')
		fourcc_mask = cv.VideoWriter_fourcc(*'XVID')
		path_img = '../test_results/'+current+'_img.avi'
		path_mask = '../test_results/'+current+'_mask.avi'
		out_img = cv.VideoWriter(path_img, fourcc_img, 20.0, (640,480))
		out_mask = cv.VideoWriter(path_mask, fourcc_mask, 20.0, (640,480), 0)

	if flag:
		out_img.write(img)
		out_mask.write(edges)

	key = cv.waitKey(1) & 0xFF
	if key == ord('s'):
		cv.imwrite('test_results/intersection.jpg',img)
	elif key == ord('q'):
		break
	elif key == ord('v'):
		SAVE_VIDEO = True

cap.release()
out_img.release()
out_mask.release()
cv.destroyAllWindows()
