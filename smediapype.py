from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# define a video capture object
vid = cv2.VideoCapture(1)
vid.set(3,515)
vid.set(4,515)

while(True):
      
    #frame
    ret, frame = vid.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results =  mp_hands.Hands().process(frame)
    landmarks = results.multi_hand_landmarks
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,hand_landmarks,connections=mp_hands.HAND_CONNECTIONS)
        points = np.asarray(landmarks[0], landmarks[5], landmarks[17])
        normal_vector = np.cross(points[2] - points[0], points[1] - points[2])
        normal_vector /= np.linalg.norm(normal_vector)
        print(normal_vector)
 
    cv2.imshow('frame', frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()