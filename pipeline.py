import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def get_calibration_data():
    # Read in the list of calibration images
    images = glob.glob("./camera_cal/calibration*.jpg")
    print(len(images))

    objpoints = []  # 3D points coordinates in real world space
    imgpoints = []  # 2D points coordinates in image place

    # Generate objpoints
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Traverse calibration images, fill imgpoints[]
    for fname in images:
        img = cv.imread(fname)

        # convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

        if ret is True:
            # print("true ", index)
            imgpoints.append(corners)
            objpoints.append(objp)

    return objpoints, imgpoints


def calc_undistort(img, objpoints, imgpoints):

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Use objpoints & imgpoints to calculate calibration matrix and other coefficients
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Undistort the image
    undist = cv.undistort(img, mtx, dist, None)
    return undist


# Get calibration data first
objpts, imgpts = get_calibration_data()

# 1. calculate calibration matrix and coefficients
# 2. Apply a distortion correction to raw images.
img = cv.imread("./camera_cal/calibration1.jpg")

undistorted = calc_undistort(img, objpts, imgpts)

# draw an output image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.savefig("./output_images/undistort.png")
plt.show()


