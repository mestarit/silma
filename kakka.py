import cv2
import numpy as np
import time
# initialize the camera
cap = cv2.VideoCapture(1)

# set the frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# set up the stereo block matching algorithm
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# set up the face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
    # capture a frame from the camera
    ret, frame = cap.read()
    
    # split the frame into left and right frames
    left_frame = frame[:, :frame_width//2, :]
    right_frame = frame[:, frame_width//2:, :]
    
    # convert the frames to grayscale
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    
    # perform stereo matching
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left_gray, right_gray)

    # detect faces
    faces = face_cascade.detectMultiScale(left_gray, 1.3, 5)
    focal_length = 2.8
    baseline = 6

    # calculate the depth for each face and put text with the depth over it using the stereo parameters
    for (x, y, w, h) in faces:
        Z = focal_length * baseline / disparity[y + h // 2, x + w // 2]
        cv2.putText(left_frame, '%.1fcm' % (Z / 10), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        cv2.rectangle(left_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # display the disparity
    cv2.imshow('disparity', disparity)

    # display the left frame
    cv2.imshow('frame', left_frame)

    
    # wait for a key press
    if cv2.waitKey(1) == ord('q'):
        break



# release the camera
cap.release()

# close all windows
cv2.destroyAllWindows()