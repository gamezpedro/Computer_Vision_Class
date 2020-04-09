import numpy as np
import cv2

click = False
pixel = [0,0,0]
pixel_hsv = [0,0,0]
lower_color = np.array([0, 0, 0])
upper_color = np.array([0, 0, 0])
kernel = np.ones((10,10), np.uint8)


def pixel_mouse(event,x,y,flags,param):
    global pixel, lower_color, upper_color, click
    if event == cv2.EVENT_LBUTTONUP:
        pixel = frame[y,x]
        lower_color = np.array([pixel[0]-10, 50, 50])
        upper_color = np.array([pixel[0]+10, 255, 255])
        '''
        print('pixel[0]-25: ',pixel[0]-25)
        print('pixel[0]+25: ',pixel[0]+25)
        print('value: ',pixel[0])
        print('lower: ',lower_color)
        print('upper: ',upper_color)
        '''
        click = True

video = cv2.VideoCapture(0)

cv2.namedWindow('video')
cv2.setMouseCallback('video',pixel_mouse)

while(video.isOpened()):
    _, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('video', frame)

    if click:
        mask = cv2.inRange(hsv, lower_color, upper_color)
        res = cv2.bitwise_and(frame, frame, mask = mask)
        #opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('closing',closing)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()