import pyrealsense2 as rs

import cv2
import numpy as np 

import time

np.seterr(over='ignore')

# Configure stream
pipeline  = rs.pipeline()
config    = rs.config()
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8,   30)
config.enable_stream(rs.stream.color,    	1280, 720, rs.format.rgb8, 30)

# Start input streaming
pipeline.start(config)
# Disable laser emitter
ir_sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
ir_sensor.set_option(rs.option.emitter_enabled, 0)
ir_sensor.set_option(rs.option.enable_auto_exposure, 0)
#ir_sensor.set_option(rs.option.exposure, 40000)
rgb_sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
rgb_sensor.set_option(rs.option.enable_auto_white_balance, 0)
rgb_sensor.set_option(rs.option.enable_auto_exposure, 0)

for i in range(30):
	frames = pipeline.wait_for_frames()


#IR exposure value: 15000
#RGB exposure value: 195

TARGET = 169

def gradient_descent(value_array):
	global TARGET

	x0, y0 = value_array[0].copy()
	x1, y1 = value_array[1].copy()

	if ((y1-y0) == 0) | ((x1-x0)==0):
		value_array[1][0] = x1 +  np.random.randint(1, 20) #Some stochastic thing
	else:
		new_x  = round(x1 - (y1-TARGET)/((y1-y0)/(x1-x0)))
		value_array[0]    = value_array[1].copy()
		value_array[1][0] = new_x
	return value_array

ir_array  = [[0,0], [1,0]]
rgb_array = [[0,0], [1,0]]

init = True
t0 = time.time()
try:
	while True:
		if ((time.time() - t0) < 0.3):

			frames     = pipeline.wait_for_frames()
			#ir_frames  = frames.get_infrared_frame()
			rgb_frames = frames.get_color_frame()

			#ir_frame   = np.asanyarray(ir_frames.get_data())
			rgb_frame  = np.asanyarray(rgb_frames.get_data()) 
			gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

			#cv2.imshow('ir_frame',   cv2.resize(ir_frame, (0,0), fx=0.4, fy=0.4))
			cv2.imshow('gray_frame', cv2.resize(gray_frame, (0,0), fx=0.4, fy=0.4))
			c = cv2.waitKey(1) % 0x100
			if c == 27:
				break

			#y_ir   = cv2.resize(ir_frame, (1,1))[0][0]
			y_gray = cv2.resize(gray_frame, (1,1))[0][0]

		else:
			#ir_array[1][1]  = y_ir
			rgb_array[1][1] = y_gray
			print('Exposure:  ' + str(rgb_array[1][0]) + \
				  '| Brightness:  ' + str(rgb_array[1][1]))

			if (y_gray == TARGET):
				break
			#x_ir  = gradient_descent(ir_array)
			rgb_array = gradient_descent(rgb_array)

			#ir_sensor.set_option(rs.option.exposure,  ir_array[1][0])
			rgb_sensor.set_option(rs.option.exposure, rgb_array[1][0])

			t0 = time.time()
		
finally:
	pipeline.stop()
	cv2.destroyAllWindows()