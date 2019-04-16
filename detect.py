import cv2
import numpy as np

faces=cv2.CascadeClassifier(r"C:\Users\Sachin Koradiya\Downloads\opencv\haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('C:\Users\Milan\yolo.yml')
id=0

#font=cv2.cv.InitFont(cv2.cv.cv2.FONT_HERSHEY_SIMPLEX,1,1,0,1)
font = cv2.FONT_HERSHEY_SIMPLEX
#print(type(font))
sampleno=0
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=faces.detectMultiScale(gray,1.3,2)
    for (x,y,w,h) in face:
        sampleno=sampleno+1
        #cv2.imwrite(r"C:\Users\Sachin Koradiya\Downloads\opencv\photo\."+str(id)+"."+str(sampleno)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if id==3:
            id="milu"

        elif id==26:
            id="sachin"
        cv2.putText(img,str(id), (x,y+h), font,2, 255, 2)

        #cv2.putText(img,(x,y+h),str(id),font,(0,255,9),2)
        #cv2.waitKey(100)
    cv2.imshow("faces",img)
    if(cv2.waitKey(1)==ord('q')):
    #if(sampleno>20):
        break
cam.release()
cv2.destroyAllWindows()





















