import pyrealsense2 as rs
import numpy as np
import cv2

# Configure stream
pipeline  = rs.pipeline()
config    = rs.config()
config.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8,   30)
config.enable_stream(rs.stream.color,    1280, 720, rs.format.rgb8, 30)

# Start input streaming
pipeline.start(config)
# Disable laser emitter
ir_sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
ir_sensor.set_option(rs.option.emitter_enabled, 0)
#ir_sensor.set_option(rs.option.enable_auto_exposure, 0)
ir_sensor.set_option(rs.option.exposure, 100)
rgb_sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
#rgb_sensor.set_option(rs.option.enable_auto_exposure, 0)
rgb_sensor.set_option(rs.option.enable_auto_white_balance, 0)
#rgb_sensor.set_option(rs.option.exposure, 10)

# Ignore first 1sec for camera warm-up
for i in range(30):
    frames = pipeline.wait_for_frames()

try:
    while True:

        # Read frames
        frames    = pipeline.wait_for_frames()
        ir_frame  = frames.get_infrared_frame()
        rgb_frame = frames.get_color_frame()

        ir_intrin  = ir_frame.profile.as_video_stream_profile().intrinsics
        #print(ir_intrin.fx, ir_intrin.fy) 
        rgb_intrin = rgb_frame.profile.as_video_stream_profile().intrinsics
        #print(rgb_intrin.fx, rgb_intrin.fy) 

        # Convert to numpy image
        ir_image   = np.asanyarray(ir_frame.get_data())
        rgb_image  = np.asanyarray(rgb_frame.get_data()) 

        # Resize to get the same scale as FOV of 2 sensors are different
        center  = np.array(ir_image.shape) / 2
        w       = 640*ir_intrin.fx/rgb_intrin.fx
        h       = 480*ir_intrin.fy/rgb_intrin.fy      
        crop_ir = ir_image[int(center[0]-h/2):int(center[0]+h/2), \
                           int(center[1]-w/2):int(center[1]+w/2)]
        fov_equalized_ir = cv2.resize(crop_ir, (640,480))

        # Lower quality of RGB image to match with cropped IR
        resized_rgb = cv2.resize(rgb_image, (0,0), fx=ir_intrin.fx/rgb_intrin.fx, \
                                     fy=ir_intrin.fy/rgb_intrin.fy)
        lower_qual_rgb = cv2.resize(rgb_image, (640, 480))

        # Gray RGB
        gray_rgb = cv2.cvtColor(lower_qual_rgb, cv2.COLOR_BGR2GRAY)
        #Convert RGB and IR to threahold binary
        
        '''
        gray_rgb_th = cv2.adaptiveThreshold(gray_rgb,255,\
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                            cv2.THRESH_BINARY,11,2)
        ir_th       = cv2.adaptiveThreshold(fov_equalized_ir,255,\
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                            cv2.THRESH_BINARY,11,2)
        '''

        _, gray_rgb_th = cv2.threshold(gray_rgb, 127, 255, cv2.THRESH_BINARY)
        _, ir_th       = cv2.threshold(fov_equalized_ir, 127, 255, \
                                       cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        eroded_gray_rgb = cv2.erode(gray_rgb_th, kernel, iterations = 50)

        # Subtract Gray RGB with IR
        subtracted = cv2.subtract(ir_th, eroded_gray_rgb)

        # Display image
        cv2.imshow('test1', gray_rgb)
        cv2.imshow('test2', fov_equalized_ir)
        cv2.imshow('test3', subtracted)

        # Exit on ESC key
        c = cv2.waitKey(1) % 0x100
        if c == 27:
            break

finally:
    pipeline.stop() 
    cv2.destroyAllWindows()