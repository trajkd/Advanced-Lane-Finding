import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip

def camera_calibration():
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Arrays to store object points and image points for all the images
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...., (8,5,0)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates

    for fname in images:
        img = mpimg.imread(fname)
        # Convert undistorted image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    return mtx, dist

def perspective_transform(img):
	 # Define calibration box in source (original) and destination (desired or warped) coordinates
	 img_size = (img.shape[1], img.shape[0])

	 # Four source coordinates
	 src = np.float32(
	 	[[595, 450],
	 	 [690, 450],
	 	 [195, 720],
	 	 [1120, 720]])

	 # Four desired coordinates
	 dst = np.float32(
	 	[[195, 0],
	 	 [1120, 0],
	 	 [195, 720],
	 	 [1120, 720]])

	 # Compute the perspective transform, M
	 M = cv2.getPerspectiveTransform(src, dst)
	 Minv = cv2.getPerspectiveTransform(dst, src)

	 # Create warped image - uses linear interpolation
	 warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

	 return warped, Minv

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/900 # meters per pixel in x dimension

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Apply threshold
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction

	# Take Sobel x and y gradients
	sobel_kernel = 15
	sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction, 
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	dir_binary =  np.zeros_like(absgraddir)
	# Apply threshold
	dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
	return dir_binary

def process_image(image):
	#image = mpimg.imread("test_images/"+os.listdir("test_images/")[2])
	# Undistort image
	undist = cv2.undistort(image, mtx, dist, None, mtx)
	# Convert to HLS color space and take the S channel
	hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	# Choose a Sobel kernel size
	ksize = 3 # Choose a larger odd number to smooth gradient measurements

	# Apply each of the thresholding functions
	gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
	grady = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=ksize, thresh=(20, 100))
	mag_binary = mag_thresh(s_channel, sobel_kernel=ksize, mag_thresh=(30, 100))
	dir_binary = dir_threshold(s_channel, sobel_kernel=ksize, thresh=(0.7, 1.3))

	# Select for pixels where both the x and y gradients meet the threshold criteria, 
	# or the gradient magnitude and direction are both within their threshold
	combined = np.zeros_like(dir_binary)
	combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
	binary_warped, Minv = perspective_transform(combined)

	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]//2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# HYPERPARAMETERS
	# Choose the number of sliding windows
	nwindows = 9
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50

	# Set height of windows - based on nwindows above and image shape
	window_height = np.int(binary_warped.shape[0]//nwindows)
	# Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated later for each window in nwindows
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),
	    (win_xleft_high,win_y_high),(0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),
	    (win_xright_high,win_y_high),(0,255,0), 2) 
	    
	    # Identify the nonzero pixels in x and y within the window #
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
	    
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices (previously was a list of lists of pixels)
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each using `np.polyfit`
	if lefty.size > 0 and leftx.size > 0:
		left_fit = np.polyfit(lefty, leftx, 2)
	else:
		return binary_warped
	if righty.size > 0 and rightx.size > 0:
		right_fit = np.polyfit(righty, rightx, 2)
	else:
		return binary_warped

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	try:
	    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	except TypeError:
	    # Avoids an error if `left` and `right_fit` are still none or incorrect
	    print('The function failed to fit a line!')
	    left_fitx = 1*ploty**2 + 1*ploty
	    right_fitx = 1*ploty**2 + 1*ploty

	# out_img[lefty, leftx] = [255, 0, 0]
	# out_img[righty, rightx] = [0, 0, 255]

	# plt.imshow(out_img)
	# plt.plot(left_fitx, ploty, color='yellow')
	# plt.plot(right_fitx, ploty, color='yellow')
	# plt.xlim(0, 1280)
	# plt.ylim(720, 0)

	## Search around polynomial
	# HYPERPARAMETER
	# Choose the width of the margin around the previous polynomial to search
	margin = 100

	# Grab activated pixels
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	### Set the area of search based on activated x-values ###
	### within the +/- margin of our polynomial function ###
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
	                left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
	                left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
	                right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
	                right_fit[1]*nonzeroy + right_fit[2] + margin)))

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit new polynomials
	left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

	## Visualization ##
	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
	                          ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
	                          ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	# Plot the polynomial lines onto the image
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')

	# plt.imshow(result)
	# plt.show()
	## End visualization steps ##

	# Calculate the curvature of polynomial functions in pixels.
	# Define y-value where we want radius of curvature
	# We'll choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)

	# Calculation of R_curve (radius of curvature)
	left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	return result

#img = mpimg.imread('output_images/frame_from_project_video.jpg')
#img = mpimg.imread('test_images/straight_lines1.jpg')
mtx, dist = camera_calibration()
#process_image(img)
output = 'project_video_result.mp4'
clip = VideoFileClip("project_video.mp4")
white_clip = clip.fl_image(process_image)
white_clip.write_videofile(output, audio=False)