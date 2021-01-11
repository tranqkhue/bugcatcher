import pyrealsense2 as rs

import cv2
import numpy as np

import time

cv2.ocl.setUseOpenCL(True)

#=================================================================================

def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """
       
    # Detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        #descriptor = cv2.xfeatures2d.SURF_create()
        descriptor = cv2.cuda.SIFT_SURF_create(300,_nOctaveLayers=2)
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # Get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)

#=================================================================================

def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

#=================================================================================
  
def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

#=================================================================================

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            							 reprojThresh)

        return (matches, H, status)
    else:
        return None

#=================================================================================

# Configure stream
pipeline  = rs.pipeline()
config    = rs.config()
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8,   30)
config.enable_stream(rs.stream.color,    	1280, 720, rs.format.rgb8, 30)

#---------------------------------------------------------------------------------

# Start input streaming
pipeline.start(config)
# Disable laser emitter
ir_sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
ir_sensor.set_option(rs.option.emitter_enabled, 0)
ir_sensor.set_option(rs.option.enable_auto_exposure, 1)
#ir_sensor.set_option(rs.option.exposure, 40000)
rgb_sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
rgb_sensor.set_option(rs.option.enable_auto_white_balance, 0)
rgb_sensor.set_option(rs.option.enable_auto_exposure, 1)
#rgb_sensor.set_option(rs.option.exposure, 400)

#---------------------------------------------------------------------------------
FEATURE_EXTRACTOR = 'sift'
CONNECTIVITY = 8
DRAW_CIRCLE_RADIUS = 4

# Ignore first 1sec for camera warm-up
for i in range(30):
	frames = pipeline.wait_for_frames()
try:
	while True:
		frames     = pipeline.wait_for_frames()
		ir_frames  = frames.get_infrared_frame()
		rgb_frames = frames.get_color_frame()

		ir_intrin  = ir_frames.profile.as_video_stream_profile().intrinsics
        #print(ir_intrin.fx, ir_intrin.fy) 
		rgb_intrin = rgb_frames.profile.as_video_stream_profile().intrinsics
        #print(rgb_intrin.fx, rgb_intrin.fy) 

		ir_frame   = np.asanyarray(ir_frames.get_data())
		rgb_frame  = np.asanyarray(rgb_frames.get_data()) 
		gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

        # Resize to get the same scale as FOV of 2 sensors are different
		center  = np.array(ir_frame.shape) / 2
		w       = 1280*ir_intrin.fx/rgb_intrin.fx
		h       = 720*ir_intrin.fy/rgb_intrin.fy      
		crop_ir = ir_frame[int(center[0]-h/2):int(center[0]+h/2), \
						   int(center[1]-w/2):int(center[1]+w/2)]
		ir_frame 	= cv2.resize(crop_ir, 	 (640,480))
		gray_frame 	= cv2.resize(gray_frame, (640,480)) 

		t0 = time.time()
		kpsA, featuresA = detectAndDescribe(ir_frame,   method=FEATURE_EXTRACTOR)
		kpsB, featuresB = detectAndDescribe(gray_frame,	method=FEATURE_EXTRACTOR)
		print(time.time() - t0)

		matches   = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, \
	    							  method=FEATURE_EXTRACTOR)
		match_viz = cv2.drawMatches(ir_frame, kpsA, gray_frame, kpsB, \
	    					   		np.random.choice(matches,100),    \
	                           		None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		match_viz = cv2.resize(match_viz, (0,0), fx=0.8, fy=0.8)
		cv2.imshow('matches', match_viz)

		M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
		if M is None:
		    print("Error!")
		else:
			(matches, H, status) = M
			print(H)
			width  = ir_frame.shape[1] + gray_frame.shape[1]
			height = ir_frame.shape[0] + gray_frame.shape[0]

			aligned_ir = cv2.warpPerspective(ir_frame, H, (640,480))
			#result[0:gray_frame.shape[0], 0:gray_frame.shape[1]] = gray_frame

			subtracted = cv2.subtract(aligned_ir, gray_frame)
			cv2.imshow('warped subtracted', subtracted)

			thresh = cv2.threshold(subtracted, 200, 255, cv2.THRESH_BINARY)[1]
			thresh = cv2.erode(thresh,  None, iterations=2)
			thresh = cv2.dilate(thresh, None, iterations=4)
			cv2.imshow('threshold', thresh)

			components = cv2.connectedComponentsWithStats(thresh, CONNECTIVITY, \
														  cv2.CV_32S)

			#Draw circles around center of components
			#See connectedComponentsWithStats function for attributes of components variable
			centers = components[3][1:]
			drawn   = np.copy(cv2.resize(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),\
										(640,480)))
			#drawn   = cv2.cvtColor(drawn, cv2.COLOR_GRAY2BGR)
			for center in centers:
				cv2.circle(drawn, (int(center[0]), int(center[1])), \
						   DRAW_CIRCLE_RADIUS, (0,0,255), thickness=-1)
			# Show the target
			cv2.imshow("target_locked", drawn)

		c = cv2.waitKey(1) % 0x100
		if c == 27:
			break

finally:
	pipeline.stop() 
	cv2.destroyAllWindows()