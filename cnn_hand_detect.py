import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model


#Loading our trained tensorfow model
model = load_model('model/rps_model1.h5')

#Setting required sizeof images for model
global IMG_SIZE

IMG_SIZE = [150,150]

#Read in frames with the webcam
cap = cv2.VideoCapture(0)

#Sets the name and size of window
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame',900,700)

#Reshaping our region of interest to allow model infer on it
def reshape_roi(roi):
    roi = tf.image.resize(tf.expand_dims(roi,0),IMG_SIZE)
    return roi

#Predicting on our ROI
def predict_img(model,roi):
    prediction = model.predict(roi).argmax(axis=1)
    return prediction


#Cchecking for the hand gesture in our ROI
def check_hand(frame):
    cv2.rectangle(frame,(50,50),(270,270),color=(0,255,0),thickness=3)
    roi = frame[50:250,50:250]
    #print(roi.shape)#(200,200,3)
    roi = reshape_roi(roi)
    prediction = predict_img(model,roi)
    #print(prediction)
    if prediction == 0:
        cv2.putText(frame,'PAPER',fontFace= cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,org=(50,300),color=(0,255,0),thickness=3)
    elif prediction == 1:
        cv2.putText(frame,'ROCK',(50,300),cv2.FONT_HERSHEY_SIMPLEX,1,color=(0,255,0),thickness=3)
    elif prediction == 2:
        cv2.putText(frame,'SCISSORS',(50,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),thickness=3)
    #select region of interest


#Checking for the presence of a hand in our ROI in order to predict gesture
def check_palm(frame):
    cv2.rectangle(frame,(50,50),(270,270),color=(0,0,0),thickness=3)
    roi = frame[50:250,50:250]
    palm_detect = cv2.CascadeClassifier('haarcascades/palm.xml')
    fist_detect = cv2.CascadeClassifier('haarcascades/fist.xml')
    palm = palm_detect.detectMultiScale(roi,scaleFactor=1.2,minNeighbors=2)
    fist = fist_detect.detectMultiScale(roi,scaleFactor=1.2,minNeighbors=2)
    if len(palm) == 0 and len(fist) == 0:
        return False
    else:
        return True


while True:

    #Reading in our frames
    ret,frame = cap.read()


    if check_palm(frame):
        check_hand(frame)
    else:
        continue

    cv2.imshow('frame',frame)
    #cv2.imshow('thresh',thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
