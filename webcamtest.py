# import the opencv library 
import cv2
import numpy as np

def draw_bounding_box(frame, bbox):
    if len(bbox) == 2:
        x, y = bbox
        w, h = 100, 200
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
"""

def image_matching(template, search_region, method):
    template_resized = cv2.resize(template, (search_region.shape[1], search_region.shape[0]))

    if method == 'sum_squared_difference':
        result = np.sum(np.square(template_resized - search_region))
    elif method == 'cross_correlation':
        result = np.sum(template_resized * search_region)
    elif method == 'normalized_cross_correlation':
        result = np.sum(template_resized * search_region) / (np.sqrt(np.sum(template_resized**2)) * np.sqrt(np.sum(search_region**2)))
    else:
        raise ValueError("Invalid matching method.")

    return result


def local_exhaustive_search(frame, template, bbox, method):
    if len(bbox) == 2:
        x, y = bbox
        w, h = 100, 200
    elif len(bbox) == 4:
        x, y, w, h = bbox
    else:
        raise ValueError("Invalid bounding box format.")
    search_region = frame[y-1:y+h+1, x-1:x+w+1]

    min_score = np.inf
    best_match = (x, y)

    for i in range(-100, 100):
        for j in range(-100, 100):
            shifted_region = frame[y-1+j:y+h+1+j, x-1+i:x+w+1+i]

            if shifted_region.shape == search_region.shape:
                score = image_matching(template, shifted_region, method)
                if score < min_score:
                    min_score = score
                    best_match = (x+i, y+j)

    return best_match
    """
import cv2
import sys
 
# define a video capture object 
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
        y = 340
        width = 200
        height = 400

        ret,frame = video.read()
        bbox = (x, y, width, height)

        # Display the resulting frame 
        #cv2.imshow('frame', frame) 
        draw_bounding_box(frame, bbox)
        
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(frame, bbox)
            break
    while True:
        # Read a new frame
        ok, new_frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(new_frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(new_frame, p1, p2, (0,0,255), 2, 1)
        else :
            # Tracking failure
            cv2.putText(new_frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            cv2.rectangle(new_frame, p1, p2, (0,0,255), 2, 1)
     
 
        # Display result
        cv2.imshow("Tracking", new_frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

    # After the loop release the cap object 
    video.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()