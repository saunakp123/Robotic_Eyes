import maestro
import time

# For Windows OS, find the right COM number
# m = maestro.Controller('COM3')
# For Mac OS, find the right port
m = maestro.Controller('/dev/cu.usbmodem001801121')

# 0 increase/decrease |    support    ccw/cw
# 1 increase/decrease | left  handle  up/down
# 2 increase/decrease | right handle  up /down
# 7 increase/decrease | right eye     down/up
# 8 increase/decrease | right eye     left/right

chan = 4

# m.setRange(chan, 1400, 1600)

xmin = m.getMin(chan)
xmax = m.getMax(chan)
for chan in range(0, 24):
    pos = m.getPosition(chan)
    print(pos)

# print(xmin, xmax, pos)

# for i in range(0, 10):
#     m.setTarget(7, 5200 + i*100)
#     time.sleep(1)

# m.setTarget(2, 5500)

m.close()