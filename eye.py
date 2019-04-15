import maestro

m = maestro.Controller('COM3')

chan = 4

# m.setRange(chan, 1400, 1600)

xmin = m.getMin(chan)
xmax = m.getMax(chan)
for chan in range(0,24):
	pos = m.getPosition(chan)
	print(pos)

# print(xmin, xmax, pos)


# m.setTarget(4, 6000)
m.close()