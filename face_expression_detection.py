import cv2
import numpy as np
from keras.models import load_model


model = load_model('model_nika.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the facial landmark detector
predictor = cv2.face.createFacemarkLBF()
predictor.loadModel('lbfmodel.yaml')

cap = cv2.VideoCapture(0)

emotion_dict={0:'Angry', 1:'Happy'}

while True:
    ret,frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
# Detect facial landmarks
        _, landmarks = predictor.fit(gray, faces)

        # Get the mouth and eyes landmarks
        mouth_landmarks = landmarks[0][0][48:68]
        left_eye_landmarks = landmarks[0][0][36:42]
        right_eye_landmarks = landmarks[0][0][42:48]
    
    # Calculate the distance between the mouth landmarks
        mouth_distance = np.linalg.norm(mouth_landmarks[12] - mouth_landmarks[4])

        # Calculate the aspect ratio of the eyes
        left_eye_aspect_ratio = left_eye_aspect_ratio(left_eye_landmarks)
        right_eye_aspect_ratio = right_eye_aspect_ratio(right_eye_landmarks)
        eye_aspect_ratio_avg = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2


        # Classify the expression based on the mouth distance and eye aspect ratio
        if mouth_distance > 30 and eye_aspect_ratio_avg < 0.2:
            dominant_emotion = emotion_dict[1]  # Happy
        else:
            dominant_emotion = emotion_dict[0]  # Angry

    cv2.rectangle(frame, (x, y),(x+w, y+h), (255,0,0), 2)
    cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv2.imshow('frame', frame)

    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    img = cv2.imread(r'C:\Users\nika7\machine_learning_exercise\face_expression_happy_angry\test\happy\PublicTest_99399200.jpg')
    if img is not None:
        print(img.shape)
    
# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

   