# Robotic_Eyes
The aim of this project was to automate the calibration of an eye tracker, so that it won't be necessary to re-calibrate for every new user.

The setup consisted of- A Screen, The eye tracker (paired to the screen), a Camera (facing the screen), and a servo-controlled mechanism which mimicked human eyes and on which a laser and artificial robotic eyes could be mounted

By pointing the laser on the screen at regular intervals, pixel co-ordinates and its corresponding servo command angles were recorded and stored as a grid. Image processing algorithms were used to detect laser position on the screen. Once a map/grid was formed, regression methods were used to interpolate between the grid points and generate servo commands to point the laser at the desired point on the screen. The Camera was used for visual feedback of the laser position. Once calibrated, the laser was replaced by artificial robotic eyes to fool the eye tracker.

1. `test/`: Initial test.
   1. `camera_test.py`
   2. `mestro_test.py`
2. `dot_detection/`
   1. `dot_detection_HSV.py`: Use HSV to filter the red laser dot, result is worse than using RGB.
   2. `dot_detection_RGB.py`: Better to detect laser dot.
   3. `get_hue.py`: Transfer colors to HSV value in a picture.
3. `circle_detection/`
   1. `circle.py`: Use `HoughCircles` to detect circles. The result is good enough.
   2. `cc.py`: Use `connectedComponentsWithStats`. Result is noisy.
   3. `diff.py`: Use the difference between the background image and the circle image to find circles. It is highly sensitive to the movement since a slight change will cause the moire pattern to be totally different. 
   4. `gs_hou.py`: Use `morphologyEx` to improve the result. Haven't implemented yet, could be helpful.
   5. `hsv_cc.py`: First filter the image with HSV, then use `connectedComponentsWithStats`.
   6. `hsv_hou_cc.py`: Filter the image with HSV, then `connectedComponentsWithStats`, and pass this to `HoughCircles`.
   7. `threshold.py`: The best method right now. Pipeline: `img -> gray -> median blur -> HoughCircles`. **Possible improvement: gs_hou.**
4. `warp/`
   1. `transform.py`: To order the four points in a rectangle.
   2. `warp.py`: Use edge detection to extract the screen.
5. `calibration/'
   1. `main.py`: contains the main calibration routine
   2. `control.py`: contains functions which mainly interface with the servo
   3. `mapping.py`: contains functions which are related to generating pixel-servo map
   4. `perception.py`: contains function used by the camera to detect laser, calibration points, screen warping, etc

