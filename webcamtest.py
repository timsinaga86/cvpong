# import the opencv library 
import cv2
import numpy as np

def draw_bounding_box(frame, bbox, frame_num):
    if len(bbox) == 2:
        x, y = bbox
        w, h = 40, 40
    elif len(bbox) == 4:
        x, y, w, h = bbox
    else:
        raise ValueError("Invalid bounding box format.")
    x = max(0, x)
    y = max(0, y)
    w = min(frame.shape[1] - x, w)
    h = min(frame.shape[0] - y, h)

    frame_with_bbox = frame.copy()
    cv2.rectangle(frame_with_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('Frame with Bounding Box', frame_with_bbox)
# define a video capture object 
vid = cv2.VideoCapture(1) 
while(True): 
	
	# Capture the video frame 
	# by frame
    x = 200
    y = 340
    width = 200
    height = 400
    method = 'sum_squared_difference'

    ret,frame = vid.read()
    bbox = (x, y, width, height)
    draw_bounding_box(frame, bbox, 1)

	# Display the resulting frame 
    #cv2.imshow('frame', frame) 
	
	# the 'q' button is set as the 
	# quitting button you may use any 
	# desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        draw_bounding_box(frame, bbox, 1) 
        while (True):
            cv2.waitKey(1)

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
