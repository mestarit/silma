## open the camera and split it 

import cv2
import numpy as np

cap = cv2.VideoCapture(1)




while True:
    
        # split the camera from middle and show it in different windows at maximum resolution
        ret, frame = cap.read()
        if ret:
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
            # debug info for the faces in the window
            #cv2.putText(frame1, "Face Count: " + str(len(faces1)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.putText(frame2, "Face Count: " + str(len(faces2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # show tracking points in the windo


            for (x,y,w,h) in faces1:
                cv2.circle(frame1, (int(x+w/2), int(y+h/2)), 5, (0, 255, 0), -1)
            for (x,y,w,h) in faces2:
                cv2.circle(frame2, (int(x+w/2), int(y+h/2)), 5, (0, 255, 0), -1)
            
            
                

            

            
            


            cat = cv2.hconcat([frame1, frame2])
            cv2.imshow('frame',cat)

            


            
            

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

cap.release()
cv2.destroyAllWindows()
