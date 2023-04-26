import cv2
import numpy as np

# Define the size of the calibration pattern
pattern_size = (9, 6)

# Create a list of object points
objp = np.zeros((1, pattern_size[0] * pattern_size[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1, 2)

# Initialize arrays to hold object points and image points from all images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Capture multiple images of the calibration pattern
for i in range(num_images):
    # Capture image
    img = cv2.imread('image{}.jpg'.format(i))

    # Find the chessboard corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If found, add object points and image points to the lists
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

# Compute the camera matrix and distortion coefficients
ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print('K =\n', K)
print('D =', D.ravel())

# Optionally, check the reprojection error
mean_error = cv2.fisheye.checkCalibration(
    objpoints, imgpoints, K, D, rvecs, tvecs, gray.shape[::-1])
print("Reprojection error: {}".format(mean_error))
