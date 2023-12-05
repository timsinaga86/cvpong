import cv2
import time
import mediapipe as mp
import numpy as np

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros(21*3)
    #return np.concatenate([ lh, rh])
    return lh, rh

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(1)
 
# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0
 
while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()
 
    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))
 
    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
 
    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    # Drawing Right hand Land Marks
    mp_drawing.draw_landmarks(
      image, 
      results.right_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )
    #print(results.right_hand_landmarks)
    # Drawing Left hand Land Marks
    mp_drawing.draw_landmarks(
      image, 
      results.left_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )
    #print(results.left_hand_landmarks)
    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime

    if results:
        keypoints_l, keypoints_r = extract_keypoints(results)
        if keypoints_r[0].size > 1:
            normal_vector_r = np.cross(np.subtract(keypoints_r[17],keypoints_r[0]), np.subtract(keypoints_r[5],keypoints_r[17]))
            print("right: " + str(normal_vector_r))
        if keypoints_l[0].size > 1:
            normal_vector_l = np.cross(np.subtract(keypoints_l[17],keypoints_l[0]), np.subtract(keypoints_l[5],keypoints_l[17]))
            print("left: " + str(normal_vector_l))

    # Displaying FPS on the image
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
 
    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)
 
    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
 
# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()