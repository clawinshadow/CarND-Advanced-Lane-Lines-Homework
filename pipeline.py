import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from camera_calib import *
from hls_gradient import *
from curve_fitting import *

# debug count
count = 1

# Get calibration data first
objpts, imgpts = get_calibration_data("./camera_cal/calibration*.jpg")

# -------------------------- Task 1 ----------------------------------
# # 1. calculate calibration matrix and coefficients
# # 2. Apply a distortion correction to raw images.
# img = mpimg.imread("./camera_cal/calibration1.jpg")
# undistorted = calc_undistort(img, objpts, imgpts)
#
# # draw an output image
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=30)
# ax2.imshow(undistorted)
# ax2.set_title('Undistorted Image', fontsize=30)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#
# plt.savefig("./output_images/undistort.png")
# plt.show()

# -------------------------- Task 2 ----------------------------------

# # Pipeline: 1. Provide an example of a distortion-corrected image.
# img = mpimg.imread("./test_images/test1.jpg")
# undistorted = calc_undistort(img, objpts, imgpts)
# plt.imshow(undistorted)
# plt.imsave("./output_images/undistort_test1.png", undistorted)
#
# plt.show()

# -------------------------- Task 3 ----------------------------------

# Pipeline: 2. Use gradient and HLS color space to create a thresholded binary image
# img = mpimg.imread("./test_images/straight_lines1.jpg")
# undistorted = calc_undistort(img, objpts, imgpts)
# combo_binary, color_binary = hls_gradient_filter(undistorted)
#
# # plotting
# f, axes = plt.subplots(1, 3, figsize=(20, 12))
# axes[0].set_title('Original image')
# axes[0].imshow(img)
#
# axes[1].set_title('Stacked thresholds')
# axes[1].imshow(color_binary.astype(np.uint8))
#
# axes[2].set_title('Combined S channel and gradient thresholds')
# axes[2].imshow(combo_binary, cmap='gray')
#
# plt.imsave("./output_images/binary_straight_lines1.png", combo_binary, cmap='gray')
#
# plt.show()

# -------------------------- Task 4 ----------------------------------

# Pipeline: 3. Perform a perspective transform
#
# img = mpimg.imread("./test_images/test2.jpg")
# undistorted = calc_undistort(img, objpts, imgpts)
# warped, src, dst = warper(undistorted)
#
# # draw the polygon of source and dest. points
# src2 = np.int32(src.reshape((-1, 1, 2)))
# cv.polylines(img, [src2], True, color=[255, 0, 0], thickness=2)
# dst2 = np.int32(dst.reshape((-1, 1, 2)))
# cv.polylines(warped, [dst2], True, color=[255, 0, 0], thickness=2)
#
# # integrate into an output image
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Undistorted Image with source points drawn', fontsize=20)
# ax2.imshow(warped)
# ax2.set_title('Warped result with dest. points drawn', fontsize=20)
# plt.savefig("./output_images/warped_test2.png")
#
# plt.show()

# -------------------------- Task 5 ----------------------------------

# Pipeline: 4. identify lane-line pixels and fit their positions with a polynomial
# img = mpimg.imread("./test_images/test5.jpg")
# undistorted = calc_undistort(img, objpts, imgpts)
# combo_binary, color_binary = hls_gradient_filter(undistorted)
# warped_binary, src, dst = warper(combo_binary)
#
#
# left_fit_prior, right_fit_prior, out_img = fit_polynomial(warped_binary, 1, 1, True)
# result = search_around_poly(warped_binary, left_fit_prior, right_fit_prior, True)
#
# plt.imshow(result)
# plt.savefig("./output_images/fit_test5.png")
# plt.show()

# -------------------------- Task 6 ----------------------------------

# # Pipeline 5: calculate the radius of curvature of the lane and the offset of the vehicle
# img = mpimg.imread("./test_images/test5.jpg")
# undistorted = calc_undistort(img, objpts, imgpts)
# combo_binary, color_binary = hls_gradient_filter(undistorted)
# warped_binary, src, dst = warper(combo_binary)
#
# left_x_base, right_x_base = histogram_peaks(warped_binary)
# xm_per_pix = 3.7 / 700
# ym_per_pix = 30 / 720
# left_fit, right_fit, out_img = fit_polynomial(warped_binary, xm_per_pix, ym_per_pix)
# offset, left_radius, right_radius = calc_curvature_offset(warped_binary, left_x_base, right_x_base,
#                                                           left_fit, right_fit)
# print(offset, left_radius, right_radius)

# -------------------------- Task 7 ----------------------------------

# Pipeline 6: result plotted back down onto the road
img = mpimg.imread("./test_images/test5.jpg")
undistorted = calc_undistort(img, objpts, imgpts)
combo_binary, color_binary = hls_gradient_filter(undistorted)
warped_binary, src, dst = warper(combo_binary)

# Fit polynomial in pixel space
left_fit, right_fit, out_img = fit_polynomial(warped_binary)
# Get plot data
ploty, left_fitx, right_fitx = get_plot_data(warped_binary, left_fit, right_fit)

## Visualization ##
# 1. Fill the polygon
# Create an image to draw the lines on
warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
Minv, src, dst = get_warp_matrix(warped_binary, inv=True)
newwarp = cv.warpPerspective(color_warp, Minv, (warped_binary.shape[1], warped_binary.shape[0]))
# Combine the result with the original image
result = cv.addWeighted(undistorted, 1, newwarp, 0.3, 0)

# Add text onto the result image
left_x_base, right_x_base = histogram_peaks(warped_binary)
xm_per_pix = 3.7 / 700
ym_per_pix = 30 / 720
left_fit_cr, right_fit_cr, out_img = fit_polynomial(warped_binary, xm_per_pix, ym_per_pix)
offset, left_radius, right_radius = calc_curvature_offset(warped_binary, left_x_base, right_x_base,
                                                          left_fit_cr, right_fit_cr)
radius = (left_radius + right_radius) / 2
radiusText = str.format("Radius of Curvature = {0:.1f}(m)", radius)
if offset < 0:
    offsetText = str.format("Vehicle is {0:.2f}m left of center", abs(offset))
else:
    offsetText = str.format("Vehicle is {0:.2f}m right of center", abs(offset))
cv.putText(result, radiusText, (100, 80), cv.FONT_HERSHEY_SIMPLEX, 2, color=[255, 255, 255], thickness=2)
cv.putText(result, offsetText, (100, 130), cv.FONT_HERSHEY_SIMPLEX, 2, color=[255, 255, 255], thickness=2)

plt.imshow(result)
plt.imsave("./output_images/final_output_test5.png", result)
plt.show()

# -------------------------- Task 8 ----------------------------------

# Pipeline (video): generate the final video output

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        self.recent_xfitted = []
        self.best_xfitted = None

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        # cache the latest 20 results
        self.N = 10

    def update(self, fit_coefficients, fit_x, fit_y, offset, radius):
        # Add current fit into cache
        self.current_fit = fit_coefficients
        if self.best_fit is None:
            self.best_fit = self.current_fit
        # Validate current fit
        self.diffs = self.current_fit - self.best_fit
        self.detected = np.sum(np.absolute(self.diffs)) < 50
        print(np.sum(np.absolute(self.diffs)))
        # Update the reset members
        if self.detected:
            # Add current fitted x values into the cache list
            self.recent_xfitted.append(fit_x)
            self.recent_xfitted = self.recent_xfitted[-self.N:]
            self.best_xfitted = np.average(np.array(self.recent_xfitted), axis=0)
            # Calculate best fit
            self.best_fit = np.polyfit(fit_y, self.best_xfitted, 2)
            # Add curvature radius and vehicle offset into the cache
            self.line_base_pos = offset
            self.radius_of_curvature = radius
            # Prepare data to fill polygon
            self.allx = fit_x
            self.ally = fit_y
        else:
            # Add current fitted x values into the cache list
            self.recent_xfitted.append(fit_x)
            self.recent_xfitted = self.recent_xfitted[-self.N:]
            self.best_xfitted = np.average(np.array(self.recent_xfitted), axis=0)
            # Calculate best fit
            self.best_fit = np.polyfit(fit_y, self.best_xfitted, 2)
            # If current fit failed, use the prior valid fitted result
            self.allx = self.best_xfitted


def fit_polynomial_ex(binary_warped, xm_per_pix=1, ym_per_pix=1):
    """
    The difference from fit_polynomial() in curve_fitting.py is it use Line Class here
    """
    # Find our lane pixels first
    if left_line_cache.detected and right_line_cache.detected:
        leftx, lefty, rightx, righty = search_around_poly(binary_warped, left_line_cache.best_fit, right_line_cache.best_fit)
    else:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Get plot data
    ploty, left_fitx, right_fitx = get_plot_data(binary_warped, left_fit, right_fit)

    # DEBUG
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # # Plots the left and right polynomials on the lane lines
    # ax3.imshow(out_img)
    # ax3.plot(left_fitx, ploty, color='yellow')
    # ax3.plot(right_fitx, ploty, color='yellow')

    # Calculate curvature radius and vehicle offset
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    left_x_base, right_x_base = histogram_peaks(binary_warped)
    offset, left_radius, right_radius = calc_curvature_offset(binary_warped, left_x_base, right_x_base,
                                                              left_fit_cr, right_fit_cr)

    # Update left Line() instance
    left_line_cache.update(left_fit, left_fitx, ploty, offset, left_radius)
    # Update right Line() instance
    right_line_cache.update(right_fit, right_fitx, ploty, -offset, right_radius)


def mark_image(undistorted, warped_binary):
    # 1. Fill the polygon
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line_cache.allx, left_line_cache.ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line_cache.allx, right_line_cache.ally])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv.warpPerspective(color_warp, Minv, (warped_binary.shape[1], warped_binary.shape[0]))
    # Combine the result with the original image
    result = cv.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    # Add text onto the result image
    offset = left_line_cache.line_base_pos
    radius = (left_line_cache.radius_of_curvature + right_line_cache.radius_of_curvature) / 2
    radiusText = str.format("Radius of Curvature = {0:.1f}(m)", radius)
    if offset < 0:
        offsetText = str.format("Vehicle is {0:.2f}m left of center", abs(offset))
    else:
        offsetText = str.format("Vehicle is {0:.2f}m right of center", abs(offset))
    cv.putText(result, radiusText, (100, 50), cv.FONT_HERSHEY_SIMPLEX, 2, color=[255, 255, 255], thickness=2)
    cv.putText(result, offsetText, (100, 100), cv.FONT_HERSHEY_SIMPLEX, 2, color=[255, 255, 255], thickness=2)

    return result

def process(image):
    """
    Function to process each image of the video
    :param image: original image from the video
    :return: marked image
    """
    # DEBUG
    global count
    # img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    # fname = "./debug_images/debug_{:0d}.png".format(count)
    # cv.imwrite(fname, img)
    # count += 1
    #
    # return image

    # Step 1: undistort the original image, using camera calibration data
    undistorted = calc_undistort(image, objpts, imgpts)
    # Step 2: use a filter of gradient and hls color space together, to generate a binary threshold image
    combo_binary, color_binary = hls_gradient_filter(undistorted)


    # debug
    # # plotting
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    # ax1.set_title('Original image')
    # ax1.imshow(undistorted)
    #
    # ax2.set_title('Stacked thresholds')
    # ax2.imshow(color_binary)
    #
    # ax3.set_title('Combined S channel and gradient thresholds')

    # Step 3: use perspective transform to generate a binary image of bird's eye view
    warped_binary = cv.warpPerspective(combo_binary, M,
                                       (combo_binary.shape[1], combo_binary.shape[0]), flags=cv.INTER_LINEAR)

    # Step 4: fit a polynomial curve for each lane line, and save data into the Line() instances
    fit_polynomial_ex(warped_binary, x_meters_per_pix, y_meters_per_pix)

    # DEBUG
    # fname = "./debug_images/debug_{:0d}.png".format(count)
    # plt.savefig(fname)
    # count += 1

    # Step 5: mark the image with fit results
    result = mark_image(undistorted, warped_binary)
    return result

# Prepare data for image processing
# 1. Extract data from camera calibration
objpts, imgpts = get_calibration_data("./camera_cal/calibration*.jpg")
# 2. Define hard-code variables
x_meters_per_pix = 3.7 / 700
y_meters_per_pix = 30 / 720
img = mpimg.imread("./test_images/test5.jpg")
img_size = img.shape
src = np.float32(
    [[(img_size[1] / 2) - 55, img_size[0] / 2 + 100],
     [((img_size[1] / 6) - 10), img_size[0]],
     [(img_size[1] * 5 / 6) + 60, img_size[0]],
     [(img_size[1] / 2 + 55), img_size[0] / 2 + 100]])
dst = np.float32(
    [[(img_size[1] / 4), 0],
     [(img_size[1] / 4), img_size[0]],
     [(img_size[1] * 3 / 4), img_size[0]],
     [(img_size[1] * 3 / 4), 0]])
# 3. Get perspective transform matrix
M = cv.getPerspectiveTransform(src, dst)
Minv = cv.getPerspectiveTransform(dst, src)
# 4. Initialize 2 instance of Line() for left and right lane line, respectively
left_line_cache = Line()
right_line_cache = Line()

# Generate marked video
# white_output = 'project_video_marked.mp4'
# # clip1 = VideoFileClip("project_video.mp4").subclip(20, 28)
# clip1 = VideoFileClip("project_video.mp4")
# white_clip = clip1.fl_image(process)
# white_clip.write_videofile(white_output, audio=False)
