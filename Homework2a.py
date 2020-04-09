import numpy as np
import cv2

video = cv2.VideoCapture(0)
ret, frame = video.read()
font = cv2.FONT_HERSHEY_SIMPLEX

if ret == True:
    cv2.imshow('Homework_2', frame)
    cv2.putText(frame, 'Victor Sebastian Martinez Perez', (frame.shape[1]-370, 25), font, 0.7, (0,0,0), 2,cv2.LINE_AA)
    cv2.putText(frame, 'A01232474', (frame.shape[1]-150, 50), font, 0.7, (0,0,0), 2,cv2.LINE_AA)
    cv2.waitKey(0)
    cv2.imwrite('myImage.jpg', frame)
    

video.release()
cv2.destroyAllWindows()
