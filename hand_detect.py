import cv2
import numpy as np
from sklearn.metrics import pairwise


#Read in frames from webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame',900,700)


#def thresh_hsv(frame):
#    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#    hsv = cv2.GaussianBlur(hsv,(5,5),100)
#    return hsv






def calc_defects(roi,contours):
    #find contours based on max area
    if len(contours)==0:
        return None
    else:
        contour = max(contours,key=lambda x:cv2.contourArea(x))
        hull = cv2.convexHull(contour,returnPoints=False)
        defects = cv2.convexityDefects(contour,hull)
        print(type(defects))
        print(defects.shape)
        if type(defects) == None:
            return roi
        else:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                cv2.line(roi,start,end,[0,255,0],2)
                cv2.circle(roi,far,5,[0,0,255],-1)
    return roi


def check_hand(frame,ret):
    #draw rectangle at top left corner of frame
    cv2.rectangle(frame,(50,50),(270,270),color=(0,255,0),thickness=3)

    #grab the rectangle region on the frame for processing
    roi = frame[50:250,50:250] 

    #Blur image
    blur_roi = cv2.blur(roi,ksize=(5,5))

    #Change to grayscale
    gray_roi = cv2.cvtColor(blur_roi,cv2.COLOR_BGR2GRAY)
    

    #Image thresholding
    _,thresh = cv2.threshold(gray_roi,110,255,cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.dilate(thresh,kernel,iterations=3)
    thresh = cv2.GaussianBlur(thresh,(5,5),100)

    #finding contours of palm in image
    #thresh = thresh_hsv(blur_roi)
    
    _,contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    calc_defects(roi,contours)

    #contour = max(contours,key=lambda x:cv2.contourArea(x))
    
    #hull = cv2.convexHull(contour)
    
    #cv2.drawContours(roi,hull,-1,(0,255,0),3)
    #correct
    return thresh,frame



while True:
    ret,frame = cap.read()

    #Process frames
    thresh,frame = check_hand(frame,ret)

    cv2.imshow('frame',frame)
    cv2.imshow('thresh',thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
