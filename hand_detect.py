import cv2
import numpy as np


#Read in frames from webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame',900,700)

def check_hand(frame,ret):
    #draw rectangle at top left corner of frame
    cv2.rectangle(frame,(50,50),(270,270),color=(0,0,255),thickness=3)

    #grab the rectangle region on the frame for processing
    roi = frame[50:271,50:271] 

    #Blur image
    blur_roi = cv2.blur(roi,ksize=(5,5))

    #Change to grayscale
    gray_roi = cv2.cvtColor(blur_roi,cv2.COLOR_BGR2GRAY)

    #Image thresholding
    _,thresh = cv2.threshold(gray_roi,110,255,cv2.THRESH_BINARY)
    thresh = cv2.GaussianBlur(thresh,(5,5),100)

    #finding contours of palm in image
    
    _,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #Found contours but having issues with calculating the convexhull
    #draw contours
    cv2.drawContours(roi,contours,-1,(0,255,0),3)

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
