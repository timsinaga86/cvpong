import sys
import numpy as np
from PIL import Image

train_files = ['skin1.jpg', 'skin2.jpg', 'skin3.jpg', 'skin4.jpg', 'skin5.jpg', 'skin6.jpg']
test_files = ['gun1.bmp', 'joy1.bmp', 'pointer1.bmp']

def skin_hsv(img, threshold):
    pixels = []
    width, height = img.shape[0], img.shape[1]
    for y in range(width):
        for x in range(height):
            if img[y][x][2] > threshold:
                pixels.append(img[y][x])
    return pixels

def segmentation_hsv(img):
    pixels=[]
    for file in train_files:
        img_train = Image.open(file)
        img_train = np.array(img_train.convert('HSV'))
        pixels += skin_hsv(img_train, 75)
    pixels = np.array(pixels)

    hist = np.histogram2d(pixels[:,0], pixels[:,1], 256)[0]

    hist = hist/np.max(hist)

    new_img = np.zeros(img.shape, dtype=np.uint8)
    width, height = new_img.shape[0], new_img.shape[1]
    for y in range(width):
        for x in range(height):
            if hist[img[y][x][0]][img[y][x][1]] > 0.0015:
                new_img[y][x] = img[y][x]
    return Image.fromarray(new_img, mode="HSV")

# for file in test_files:
#     img = Image.open(file)
#     img = segmentation_hsv(np.array(img.convert('HSV'))).convert('RGB')
#     img.save('outhsv'+file)

def skin_rgb(img, threshold):
    pixels = []
    width, height = img.shape[0], img.shape[1]
    for y in range(width):
        for x in range(height):
            if img[y][x][0] > threshold and img[y][x][1] > threshold and img[y][x][2] > threshold:
                pixels.append(img[y][x])
    return pixels

def segmentation_rgb(img):
    pixels=[]
    for file in train_files:
        img_train = Image.open(file)
        img_train = np.array(img_train.convert('RGB'))
        pixels += skin_hsv(img_train, 75)
    pixels = np.array(pixels)

    hist = np.histogramdd((pixels[:,0], pixels[:,1], pixels[:, 2]), 256)[0]

    hist = hist/np.max(hist)

    new_img = np.zeros(img.shape, dtype=np.uint8)
    width, height = new_img.shape[0], new_img.shape[1]
    for y in range(width):
        for x in range(height):
            if hist[img[y][x][0]][img[y][x][1]][img[y][x][2]] > 0.0015:
                new_img[y][x] = img[y][x]
    return Image.fromarray(new_img, mode="RGB")

for file in test_files:
    img = Image.open(file)
    img = segmentation_rgb(np.array(img))
    img.save('outrgb'+file)