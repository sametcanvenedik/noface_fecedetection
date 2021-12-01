# Kütüphaneleri içe aktarılması
from os import truncate
import cv2
import mymodul as md

# Gerekli tanımlamalar
face_cascade = cv2.CascadeClassifier(md.filepath("haarcascade_frontalface_default", "xml"))
cap = cv2.VideoCapture(0)

#  Giriş
while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)
        for x,y,w,h  in face_rect:
            face= frame[y:y+h,x:x+w,:]
            blurred_face = cv2.GaussianBlur(face,ksize=(29,29), sigmaX= 7)
            frame[y:y+h,x:x+w,:] =  blurred_face[:,:,:]
        md.cshow("NoFace_filter", frame)
        md.cwait(1)
    else:
        print("Görüntü okunamadı.")
        break