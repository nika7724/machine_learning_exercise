import cv2 
print(cv2.__version__)
import numpy as np
from keras.models import load_model
from keras.preprocessing import image 
from tensorflow.keras.utils import img_to_array

model = load_model(r'C:\Users\nika7\machine_learning_exercise\model_nika.h5')
face_Detect = cv2.CascadeClassifier(r'C:\Users\nika7\machine_learning_exercise\haarcascade_frontalface_default.xml')

labels_dict={0:'ANGRY', 1:'HAPPY'}

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_Detect.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y),(x+w, y+h), (0,255,255), 2)
        roi_gray = gray[y:y+h, x:x+w] 
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = model.predict(roi)[0]
            print(prediction)
            label_idx = np.argmax(prediction)
            label = labels_dict[label_idx]            
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


