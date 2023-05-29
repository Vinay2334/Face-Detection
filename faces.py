import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

og_labels={}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://192.168.1.3:8080/video')

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
       gray,
       scaleFactor=1.5,
       minNeighbors=5,
       minSize=(30,30),
       flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,w,h) in faces:
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        #Recognize MachineL 
        id_, conf = recognizer.predict(roi_gray)
        if conf>=68:
            print(conf)
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            print(name)
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        img_item = "my_image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255,0,0) #BGR
        stroke = 2 #thick
        width = x+w
        height = y+h
        cv2.rectangle(frame, (x,y), (width, height), color, stroke)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()