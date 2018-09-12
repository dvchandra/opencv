import os
import cv2
import numpy as np
from PIL import Image
recognizer=cv2.createLBPHFaceRecognizer()
path='dataset'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for ip in imagePaths:
        faceImg=Image.open(ip).convert('L')
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(ip)[-1].split('.')[1])
        faces.append(faceNp)
        print ID
        IDs.append(ID)
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
    return IDs, faces
IDs,faces=getImagesWithID(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('recognizer/trainningdata.yml')
cv2.destroyAllWindows()

        
