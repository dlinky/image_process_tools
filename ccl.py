import cv2
import numpy as np

img = cv2.imread('cropped.jpg')
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
_, threshed = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
morphed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=2)

_, labels = cv2.connectedComponents(morphed)

labels = np.uint8(labels)
height, width = labels.shape
cv2.imshow('labels', labels)
cv2.waitKey(0)
colored = np.zeros((height, width, 3), 'uint8')
for r, row in enumerate(labels):
    for c, pixel in enumerate(row):
        colored[r, c] = (pixel*5, 255, 255)
        if pixel == 0:
            colored[r, c] = (0, 0, 0)
cv2.imshow('label', colored)
cv2.waitKey(0)
colored = cv2.cvtColor(colored, cv2.COLOR_HSV2BGR)

cv2.imshow('label', colored)
cv2.waitKey(0)