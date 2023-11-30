# import the opencv library 
import cv2
import numpy as np
import segmentation as seg
import pong
import math

train_files = ['skin1.jpg', 'skin2.jpg', 'skin3.jpg', 'skin4.jpg', 'skin5.jpg', 'skin6.jpg']

# def draw_bounding_box(frame, bbox):
#     if len(bbox) == 2:
#         x, y = bbox
#         w, h = 100, 200
#     elif len(bbox) == 4:
#         x, y, w, h = bbox
#     else:
#         raise ValueError("Invalid bounding box format.")
#     x = max(0, x)
#     y = max(0, y)
#     w = min(frame.shape[1] - x, w)
#     h = min(frame.shape[0] - y, h)

#     frame_with_bbox = frame.copy()
#     cv2.rectangle(frame_with_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     cv2.imshow('Frame with Bounding Box', frame_with_bbox)

def get_hand_angle(bbox, frame):
    img = seg.segmentation_hsv(cv2.cvtColor(frame[bbox[1]: bbox[1] + bbox[3],bbox[0]:bbox[0]+bbox[2]], cv2.COLOR_BGR2HSV), pixels)
    edge = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_HSV2BGR),150, 250)
    votes = 50
    lines = cv2.HoughLines(edge, 1, np.pi/180, votes)
    theta_avg = 0
    count = 0
    while lines is None:
        votes -= 1
        lines = cv2.HoughLines(edge, 1, np.pi/180, votes)
        if votes == 0:
            return 0
    for line in lines:
        r, theta = line[0]
        theta_avg += theta
        count += 1

    theta_avg = theta_avg/count
    return theta_avg
        


def main():
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
            tracker2 = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
    video = cv2.VideoCapture(1)
    while(True): 
        
        # Capture the video frame 
        # by frame
        x = 300
        y = 400
        width = 100
        height = 200

        ret,frame = video.read()
        bbox = (x, y, width, height)
        bbox2 = (1600, 400, width, height)
        show_frame = np.copy(frame)
        # Display the resulting frame 
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(show_frame , p1, p2, (0,0,255), 2, 1)
        p3 = (int(bbox2[0]), int(bbox2[1]))
        p4= (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
        cv2.rectangle(show_frame , p3, p4, (0,0,255), 2, 1)
        show_frame = cv2.flip(show_frame , 1)
        cv2.imshow('frame', show_frame) 
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(frame, bbox)
            ok = tracker2.init(frame, bbox2)
            break
    pong.reset()
    while True:
        # Read a new frame
        ok, new_frame = video.read()
        if not ok:
            continue
 
        # Update tracker
        ok1, bbox = tracker.update(new_frame)
        ok2, bbox2 = tracker2.update(new_frame)
 
        # Draw bounding box
        if ok1:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(new_frame, p1, p2, (0,0,255), 2, 1)
        if ok2:
            # Tracking success 
            p3 = (int(bbox2[0]), int(bbox2[1]))
            p4 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
        cv2.rectangle(new_frame, p3, p4, (0,0,255), 2, 1)

        #Check collisions
        is_col_bbox1 = pong.check_bbox(bbox)
        if is_col_bbox1:
            theta = get_hand_angle(bbox, frame)
            print(theta)
            #Calc velocity
            mag = pong.velocity_magnitude()
            vx = round(math.cos(theta)*mag)
            vy = round(math.sin(theta)*mag)
            (vx,vy)
            pong.update_velocity(vx,vy)
        is_col_bbox2 = pong.check_bbox(bbox2)
        if is_col_bbox2:
            theta = get_hand_angle(bbox2, frame)
            #Calc velocity
            mag = pong.velocity_magnitude()
            print(theta)
            vx = -round(math.cos(theta)*mag)
            vy = round(math.sin(theta)*mag)
            print(vx,vy)
            pong.update_velocity(vx,vy)


        pong.check_boundaries()
        pong.draw(new_frame)
        show_frame = cv2.flip(new_frame, 1)

     
 
        # Display result
        cv2.imshow("Pong", show_frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
        if k == ord('r'): pong.reset()

    # # After the loop release the cap object 
    video.release() 
    # # Destroy all the windows 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    pixels = seg.segmentation_train()
    pixels = np.array(pixels)
    main()