# Robotic_Eyes

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