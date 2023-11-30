import numpy as np
import cv2
import math

pong_data = [960, 540, 0.0, 0.0]
radius = 25

def update_position():
    pong_data[0] += np.round(pong_data[2])
    pong_data[1] += np.round(pong_data[3])

def update_velocity(x_val, y_val):
    pong_data[2] += x_val
    pong_data[3] += y_val

def velocity_magnitude():
    return math.sqrt(pong_data[2]**2 + pong_data[3]**2)

def draw(show_frame):
    #update coords
    update_position()
    cv2.circle(show_frame,(pong_data[0], pong_data[1]), radius, (255, 0, 0), -1)

def check_bbox(bbox):
    ball_next = (pong_data[0] += np.round(pong_data[2]), pong_data[1] += np.round(pong_data[3]))
    if (bbox[0] <= ball_next[0] and bbox[0] + bbox[2] >= ball_next[0]) and (bbox[1] <= ball_next[1] and bbox[1] + bbox[3] >= ball_next[1]):
        return True
    return False

def check_boundaries():
    ball_next = (pong_data[0] += np.round(pong_data[2]), pong_data[1] += np.round(pong_data[3]))
    if ball_next[1] <= 0 or ball_next:
        pong_data[3] *= -1


def reset():
    pong_data[2] = np.randint(-25, 26), np.randint(-25, 26)
    pong_data[3] = np.randint(-25, 26), np.randint(-25, 26)