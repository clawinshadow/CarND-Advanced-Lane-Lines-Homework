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

[image1]: ./output_images/undistort.png "Undistorted"
[image2]: ./output_images/undistort_test1.png "Road Transformed"
[image3]: ./output_images/binary_straight_lines1.png "Binary Example"
[image4]: ./output_images/warped_test2.png "Warp Example"
[image5]: ./output_images/fit_test5.png "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
---
First, the code structure of my project including 4 .py files in the root directory
1. `camera_calib.py`: relate to camera calibration, perspective transform. 
2. `hls_gradient.py`: generate thresholded binary images, using gradient-based and hls color space based methods
3. `curve_fitting.py`: take warped binary images as input, fit a polynomial curve to each lane line.
4. `pipeline.py`: the main code file for single image pipeline and the final video output, call the functions from the prior 3 code files, the codes for single image pipeline demos were commentted out, retain the last piece of code for generating the video output only.

---

### Camera Calibration

The code for this step is contained in lines 8 through 70 of the file called `pipepline.py`, in which defined 2 functions named `get_calibration_data()` and `calc_undistort()`
1. In `get_calibration_data()`, it prepares data (imgpoints & objpoints) for camera calibration. They were generated when I traverse all the images in the `camera_cal` folder. 
   * The __object points__ are the same for each iteration, because they are the (x, y, z) coordinates of the chessboard corners in the real world, thus `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image (only the `calibration1.jpg` failed). 
   * The __image points__ are different for each iteration, it represents the corners detected in each different image, so `imgpoints` will be appended with the (x, y) pixel position of each of the corners with each successful chessboard detection.
2. In `calc_undistort()` I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function, then applied this distortion correction to the test image `calibration1.jpg` using the `cv2.undistort()` function and save it into the `output_images` folder, the result as following:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

In this task, I just use the camera calibration data to undistort the image in `./test_images/test1.jpg`, code at `Task 1` section in `pipeline.py`, the distortion-corrected image saved in `./output_images/undistort_test1.png`, as following:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (all the codes in `hls_gradient.py`), the following is a brief introducation to the functions.
* __abs_sobel_thresh()__: apply the threshold based on the absolute value of x or y gradient, using Sobel operator
* __magnitude_thresh()__: threshold based on the magnitude of gradient, taking sobel_kernel size as a parameter
* __direction_thresh()__: threshold based on the direction of gradient, that's arctan(abs(y/x))
* __hls_thresh()__: convert the image to HLS color space first, then apply the threshold based on one of these 3 channels, mainly use L and S channels.
* __hls_gradient_filter()__: an integrated function to combine all the prior threshold functions, expect to generate a clear binary image.

In this task, I use a x-gradient threshold of (10, 120), a S-channel threshold of (100, 255) and a L-channel threshold of (50, 255) to generate the binary image.
The lower limit of s-channel threshold as small as 100, since in some bright images with light-color road surface, the s-channel of the lane line is very opaque, a lower limit is necessary to detect it.
The L-channel threshold is to eliminate the noise generated by the shadow of tree branches in s-channel threshold.
Code at `Task 3` section in `pipeline.py`, result image saved in `./output_images/binary_straight_lines1.png`

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 70 through 77 in the file `camera_calib.py`. It simply calls the `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` functions, the most critical part is how to choose the source and destination points for perspective transform. I chose the hard-coded source and destination points which was given in the project instructions:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

It works well in the video pipeline as expected. Codes at `Task 4` section in `pipeline.py`, result image as following: 

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For a brand new image without any prior fit knowledge, I use the sliding window method to find lane-line pixels
1. Find the starting points of left and right x coordinate, using the histogram peaks method, the most 2 peaks separated by midpoint would be the result
2. Define some hyperparameters such as windows count (9), windows width (200), minimum number of pixels to recenter window (50)
3. Iterate the windows from bottom to top, detect all the lane-line pixels, and then colorize them (left red, right blue)
4. Fit a 2nd polynomial curve line with each lane-line pixels
After we have a successful fit result, we can then using this prior knowledge to detect lane-line pixels more efficiently, using a bandwidth around the prior fit line, codes in `search_around_poly()` function, don't need to iterate the sliding windows anymore. 

Codes at `Task 5` section in `pipeline.py`, result image as following

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
