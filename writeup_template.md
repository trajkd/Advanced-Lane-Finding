## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted chessboard"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/undistort_test1_output.png "Undistorted test image"
[image4]: ./output_images/binary_combo_example.png "Binary Example"
[image5]: ./output_images/warped_straight_lines.png "Warp Example"
[image6]: ./output_images/color_fit_lines.png "Fit Visual"
[image7]: ./output_images/frame_from_project_video_result.jpg "Output"
[video1]: ./project_video_result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 9 through 30 of the file called `lanelines.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted chessboard][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Road Transformed][image2]
![Undistorted test image][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 75 through 142 in `lanelines.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![Binary Example][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform(img)`, which appears in lines 32 through 57 in the file `lanelines.py`.  The `perspective_transform(img)` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
	[[595, 450],
	 [690, 450],
	 [195, 720],
	 [1120, 720]])

dst = np.float32(
	[[195, 0],
	 [1120, 0],
	 [195, 720],
	 [1120, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 450      | 195, 0        | 
| 690, 450      | 1120, 0       |
| 195, 720      | 195, 720      |
| 1120, 720     | 1120, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warp Example][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial (in lines 143 through 307, where each step is commented) kinda like this:

![Fit Visual][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 309 through 323 in my code in `lanelines.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 325 through 344 in my code in `lanelines.py` at the end of the function `process_image(img)`.  Here is an example of my result on a test image:

![Output][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I mostly followed the steps outlined in the lessons and built on the concepts and code learned there. I started in a pessimistic mood as there were so many steps that it would be easy to incur in a lot of issues but when I got down to write the code I didn't have any particular problems along the way. Managing the code one concept at a time helped a lot with that. I think combining the thresholds was the most daunting task of all.

As suggested in the lesson, I used the saturation channel of the HLS color space in order to perform well in case of shadows and lines of different color.

My pipeline will likely fail if the lane lines are not in the camera view, such as when the car changes lanes, and on large curves.

The lane lines detection can be improved by getting information from the previous frames for example.
