import os
import numpy as np
from PIL import Image
import cv2


recogniser = cv2.face.LBPHFaceRecognizer_create()

path=r"C:\Users\Sachin Koradiya\Downloads\opencv\photo"

def getImages(path):
    imagepath=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagep in imagepath:
        faceimg=Image.open(imagep).convert('L')
        facenp=np.array(faceimg,'uint8')
        ID=int(os.path.split(imagep)[-1].split('.')[1])
        faces.append(facenp)
        ids.append(ID)
        cv2.imshow("training",facenp)
        cv2.waitKey(10)
    return ids,faces

ids,faces=getImages(path)
recogniser.train(faces,np.array(ids))
recogniser.save("trainingdat.yml")
cv2.destroyAllWindows()