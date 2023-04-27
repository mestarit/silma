import cv2
import numpy as np

# Load the stereo camera stream
cap = cv2.VideoCapture(2)

# Set the resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a stereo camera object and set the calibration parameters
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
stereo.setMinDisparity(0)
stereo.setNumDisparities(64)
stereo.setBlockSize(15)
stereo.setSpeckleWindowSize(100)
stereo.setSpeckleRange(32)
stereo.setDisp12MaxDiff(1)

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Capture a frame from the stereo camera stream
    ret, frame = cap.read()

    # Split the frame in half vertically
    height, width = frame.shape[:2]
    mid = int(width / 2)
    imgL = frame[:, :mid]
    imgR = frame[:, mid:]
    imgR_gray = imgR
    imgL_gray = imgL

    # Convert the left and right images to grayscale
    imgL_gray = cv2.cvtColor(imgL_gray, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR_gray, cv2.COLOR_BGR2GRAY)

    # Calculate the disparity map using the stereo camera
    disparity = stereo.compute(imgL_gray,imgR_gray)

    # Normalize the disparity map to a range between 0 and 255
    normalized_depth = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Detect faces in the left image
    faces = face_cascade.detectMultiScale(imgL_gray, scaleFactor=1.2, minNeighbors=5)

    # Iterate through each face and calculate the depth
    for (x,y,w,h) in faces:
        # Calculate the average disparity within the face bounding box
        face_disp = np.mean(disparity[y:y+h,x:x+w])
        
        # Calculate the depth from the disparity using the stereo camera calibration parameters
        baseline = 0.06 # in meters
        focal_length = 300 # in pixels
        depth = (baseline * focal_length) / face_disp
        
        # calculate the average of the depth and add it as text
        
       


        # Display the depth on the image
        cv2.putText(imgL, f"Depth: {depth:.2f} meters", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(imgL,(x,y),(x+w,y+h),(0,255,0),2)


    # Display the images
    cv2.imshow('Left Image', imgL)
    cv2.imshow('Right Image', imgR)
    cv2.imshow('Disparity Map', normalized_depth)
    
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the stereo camera stream and close all windows
cap.release()
cv2.destroyAllWindows()
