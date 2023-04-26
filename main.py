## open the camera and split it 

import cv2
import numpy as np

cap = cv2.VideoCapture("/dev/video2")




while True:
    if ret:
        # split the camera from middle and show it in different windows at maximum resolution
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        frame1 = frame[:, 0:640]
        frame2 = frame[:, 640:1280]

        #fix the fisheye effect
        DIM=(1280, 720)
        K = np.array([[720.0, 0.0, 640.0], [0.0, 720.0, 360.0], [0.0, 0.0, 1.0]])
        D = np.array([0.0, 0.0, 0.0, 0.0])
        DIM = (1280, 720)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

        frame1 = cv2.remap(frame1, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        frame2 = cv2.remap(frame2, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        #track the faces in these frames
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
        faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
        # show the faces in the frames
        for (x,y,w,h) in faces1:
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray1[y:y+h, x:x+w]
            roi_color = frame1[y:y+h, x:x+w]
        for (x,y,w,h) in faces2:
            cv2.rectangle(frame2,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray2[y:y+h, x:x+w]
            roi_color = frame2[y:y+h, x:x+w]
        # show the frames in one window with useful debug
        


        
        

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

cap.release()
cv2.destroyAllWindows()
