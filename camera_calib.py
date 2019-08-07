import cv2 as cv
import numpy as np
import glob


def get_calibration_data(directory):
    # Read in the list of calibration images
    images = glob.glob(directory)

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
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Use objpoints & imgpoints to calculate calibration matrix and other coefficients
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Undistort the image
    undist = cv.undistort(img, mtx, dist, None)
    return undist


def get_warp_matrix(img, inv=False):
    # hard-code src and dst points
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

    if inv is True:
        M = cv.getPerspectiveTransform(dst, src)
    else:
        M = cv.getPerspectiveTransform(src, dst)

    return M, src, dst


def warper(img):
    """
    Perform a perspective transform
    """
    M, src, dst = get_warp_matrix(img)
    warped = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR)
    return warped, src, dst
