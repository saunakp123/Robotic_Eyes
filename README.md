# Robotic_Eyes

1. `test_camera.py` : for testing the camera
2. `get_color.py`, `get_gray.py`, `get_hue.py` : for statically analyzing the image with color, grayscale and hue
3. `dot_detection.py` : get frame from the camera, find the pixels in certain HSV range and take their mean to get the laser dot position
4. `dot_detection2.py` : same as before but use RGB value to make it easier to test.
5. `multidots_detection.py` : use HoughCircles function in OpenCV to detect calibration circles in the image
6. `dot_intersection.py` : detect the intersection of laser dot with calibration circles and a way to save video.