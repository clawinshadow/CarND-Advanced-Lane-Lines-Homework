import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    """
    Apply a threshold on x or y gradient, using Sobel operator
    """
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Get the absolute value of x or y gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1))

    # Rescale to [0, 255]
    scaled_sobel = np.uint8(abs_sobel * 255 / np.max(abs_sobel))

    # Apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    mask = (scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])
    binary_output[mask] = 1

    return binary_output


def magnitude_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """
    Apply a threshold on the magnitude of gradient, using Sobel operator
    :param sobel_kernel: kernel size of Sobel operator
    """
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Calculate the magnitude of gradient
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

    scaled_sobel = np.uint8(sobel * 255 / np.max(sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output


def direction_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Apply a threshold on the direction of gradient, arctan(grad_y/grad_x)
    """
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Calculate the direction of the gradient
    direction_grad = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(gray)
    mask = (direction_grad >= thresh[0]) & (direction_grad <= thresh[1])
    binary_output[mask] = 1

    return binary_output


def hls_thresh(img, channel='s', thresh=(0, 255)):
    """
    Apply a threshold on the HLS color space of image
    """
    # Convert to HLS color space
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)

    # debug
    # plt.imshow(hls[:, :, 1], cmap='gray')

    # Select channel
    if channel == 'h':
        channel_data = hls[:, :, 0]
    if channel == 'l':
        channel_data = hls[:, :, 1]
    if channel == 's':
        channel_data = hls[:, :, 2]

    # Apply the threshold
    binary_output = np.zeros_like(channel_data)
    mask = (channel_data >= thresh[0]) & (channel_data <= thresh[1])
    binary_output[mask] = 1

    return binary_output

def hls_gradient_filter(img):
    # Apply x-gradient filter
    sobel_x_binary = abs_sobel_thresh(img, orient='x', thresh=(10, 120))

    # Apply y-gradient filter
    sobel_y_binary = abs_sobel_thresh(img, orient='y', thresh=(0, 255))

    # Apply gradient direction filter
    # direction_binary = direction_thresh(img, 3, thresh=(np.pi/6, np.pi/2))

    # Combine gradient-based filter
    grad_binary = np.zeros_like(sobel_x_binary)
    grad_binary[(sobel_y_binary == 1) & (sobel_x_binary == 1)] = 1

    # Apply s-channel filter in hls color space
    s_binary = hls_thresh(img, channel='s', thresh=(150, 255))

    # Apply l-channel filter in hls color space
    l_binary = hls_thresh(img, channel='l', thresh=(50, 255))

    # Combine hls-based filter
    hls_binary = np.zeros_like(s_binary)
    hls_binary[(s_binary == 1) & (l_binary == 1)] = 1

    # Combine hls-based & gradient-based filter
    combined_binary = np.zeros_like(hls_binary)
    combined_binary[(grad_binary == 1) | (hls_binary == 1)] = 1

    # Visualize it
    # Green represents sobel_x_binary = 1, Blue represents s_binary = 1
    color_binary = np.dstack((np.zeros_like(hls_binary), grad_binary, hls_binary)) * 255


    # plotting
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    # ax1.set_title('Original image')
    # ax1.imshow(img)
    #
    # ax2.set_title('Stacked thresholds')
    # ax2.imshow(color_binary)
    #
    # ax3.set_title('Combined S channel and gradient thresholds')
    # ax3.imshow(combined_binary, cmap='gray')

    return combined_binary