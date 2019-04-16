

import cv2
import numpy as np

faces=cv2.CascadeClassifier(r"C:\Users\Sachin Koradiya\Downloads\opencv\haarcascade_frontalface_default.xml)
cam=cv2.VideoCapture(0)
#rec=cv2.face.LBPHFaceRecognizer_create()
#rec.load('trainingdat.yml')
id=input("enter id")
#font=cv2.cv.InitFont(cv2.cv.CV_F)
sampleno=0
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=faces.detectMultiScale(gray,1.3,2)
    for (x,y,w,h) in face:
        sampleno=sampleno+1
        cv2.imwrite(r"C:\Users\Milan\Downloads\opencv\photo\."+str(id)+"."+str(sampleno)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #id,conf=rec.predict(gray[y:y+h,x:x+w])
        #cv2.cv.putText(cv2.cv.fromarray(img),str(id))
        cv2.waitKey(100)
    cv2.imshow("faces",img)
    cv2.waitKey(1)
    if(sampleno>20):
        break
cam.release()
cv2.destroyAllWindows()