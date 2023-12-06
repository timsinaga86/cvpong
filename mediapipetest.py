# import the opencv library 
import cv2
import numpy as np
import segmentation as seg
import pong
import math
import mediapipe as mp
import smediapype as smp


def get_hand_angle(normal_vector):
    theta_avg = np.arctan(normal_vector[1]/normal_vector[0])
    print(theta_avg)
    return theta_avg


def main():
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    normal_vector_r = [0,0]
    normal_vector_l = [0,0]
    mp_holistic = mp.solutions.holistic
    holistic_model = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initializing the drawing utils for drawing the facial landmarks on image
    mp_drawing = mp.solutions.drawing_utils

    #capture = cv2.VideoCapture(1)
    
    # Initializing current time and precious time for calculating the FPS
    video = cv2.VideoCapture(1)
    # while(True):
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # print("before pong reset")
    pong.reset()
    x = 300
    y = 400
    width = 100
    height = 200
    bbox = [x, y, width, height]
    bbox2 = [1600, 400, width, height]
    r_avg = [0,0]
    l_avg = [0,0]
    count = 0
    while True:
        # Read a new frame
        ok, new_frame = video.read()
        if not ok:
            continue
        # Draw bounding box
        # resizing the frame for better view
        #frame = cv2.resize(frame, (800, 600))
    
        # Converting the from BGR to RGB
        image = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    
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

        if results:
            keypoints_l, keypoints_r = smp.extract_keypoints(results)
            if keypoints_r[0].size > 1:
                normal_vector_r = np.cross(np.subtract(keypoints_r[17],keypoints_r[0]), np.subtract(keypoints_r[5],keypoints_r[17]))
                #print("right: " + str(normal_vector_r))
                r_avg = [np.mean([keypoints_r[0,0],keypoints_r[5,0],keypoints_r[17,0]]),np.mean([keypoints_r[0,1],keypoints_r[5,1],keypoints_r[17,1]])]
            if keypoints_l[0].size > 1:
                normal_vector_l = np.cross(np.subtract(keypoints_l[17],keypoints_l[0]), np.subtract(keypoints_l[5],keypoints_l[17]))
                l_avg = [np.mean([keypoints_l[0,0],keypoints_l[5,0],keypoints_l[17,0]]),np.mean([keypoints_l[0,1],keypoints_l[5,1],keypoints_l[17,1]])]
                #print("left: " + str(normal_vector_l))
        bbox[0] = (r_avg[0] * 1920) - width/2
        bbox[1] = (r_avg[1] * 1080) - height/2
        bbox2[0] = (l_avg[0] * 1920) - width/2
        bbox2[1] = (l_avg[1] * 1080) - height/2
        #print(bbox)
        if normal_vector_r is not None:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(image, p1, p2, (0,0,255), 2, 1)
        if normal_vector_l is not None:
            # Tracking success 
            p3 = (int(bbox2[0]), int(bbox2[1]))
            p4 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
        cv2.rectangle(image, p3, p4, (0,0,255), 2, 1)

        #Check collisions
        is_col_bbox1 = pong.check_bbox(bbox)
        if is_col_bbox1:
            theta = get_hand_angle(normal_vector_r)
            if theta > math.pi/2: theta+= math.pi
            #print(theta)
            #Calc velocity
            mag = pong.velocity_magnitude()
            vx = round(math.cos(theta)*mag)
            vy = round(math.sin(theta)*mag)
            pong.update_velocity(vx,vy)
        is_col_bbox2 = pong.check_bbox(bbox2)
        if is_col_bbox2:
            theta = get_hand_angle(normal_vector_l)
            if theta > math.pi/2: theta+= math.pi
            #Calc velocity
            mag = pong.velocity_magnitude()
            #print(theta)
            vx = -round(math.cos(theta)*mag)
            vy = round(math.sin(theta)*mag)
            pong.update_velocity(vx,vy)


        pong.check_boundaries()
        pong.draw(image)
        image = cv2.flip(image, 1)

        # Display result
        cv2.imshow("Pong", image)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
        if k == ord('r'): pong.reset()
        if k == ord('s'): 
            cv2.imwrite("test"+str(count)+".png", image)
            count += 1



    # # After the loop release the cap object 
    video.release() 
    # # Destroy all the windows 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()