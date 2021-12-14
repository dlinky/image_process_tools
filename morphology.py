import cv2
import numpy as np
import os

def empty(self):
    pass


img = cv2.imread('cropped.jpg')


windowName = 'threshold'
cv2.namedWindow(windowName)
cv2.createTrackbar('kernel size', windowName, 0, 5, empty)
cv2.createTrackbar('iteration', windowName, 0, 10, empty)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
_, threshed = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

while True:
    if cv2.waitKey(10) == 27:
        break
    kernelSize = cv2.getTrackbarPos('kernel size', windowName)
    if kernelSize == 0:
        kernelSize += 1
    iteration = cv2.getTrackbarPos('iteration', windowName)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    temp = threshed.copy()
    eroded = cv2.morphologyEx(threshed, cv2.MORPH_ERODE, kernel, iterations=iteration)
    eroded = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
    cv2.putText(eroded, 'Erode', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    dilated = cv2.morphologyEx(threshed, cv2.MORPH_DILATE, kernel, iterations=iteration)
    dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    cv2.putText(dilated, 'Dilate', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    opening = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel, iterations=iteration)
    opening = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    cv2.putText(opening, 'Open', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
    cv2.putText(temp, 'Original', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    stack1 = np.hstack((temp, eroded))
    stack2 = np.hstack((dilated, opening))
    stacked = np.vstack((stack1, stack2))

    cv2.imshow(windowName, stacked)