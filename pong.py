import numpy as np

pong_data = [960, 540, 0.0, 0.0]
radius = 25

def draw(show_frame)
    #update coords
    pong_data[0] += np.round(pong_data[2])
    pong_data[1] += np.round(pong_data[3])
    cv2.circle(frame,(pong_data[0], pong_data[1]), radius, (255, 0, 0), -1)

def check_collisions()
    return

def reset()
    pong_data = [960, 540, np.randint(-25, 26), np.randint(-25, 26)]