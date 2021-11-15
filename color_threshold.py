import copy

import cv2
import os

import numpy as np

file = [_ for _ in os.listdir() if _.endswith('.jpg')]
img = cv2.imread(file[0])


def empty(self):
    pass


windowName = 'Threshold'
cv2.namedWindow(windowName)
cv2.createTrackbar('Manual', windowName, 0, 1, empty)
cv2.createTrackbar('L_inv', windowName, 0, 1, empty)
cv2.createTrackbar('L_th', windowName, 0, 255, empty)
cv2.createTrackbar('a_inv', windowName, 0, 1, empty)
cv2.createTrackbar('a_th', windowName, 0, 255, empty)
cv2.createTrackbar('b_inv', windowName, 0, 1, empty)
cv2.createTrackbar('b_th', windowName, 0, 255, empty)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

while True:
    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()
        break

    temp = copy.deepcopy(img)
    manual_switch = cv2.getTrackbarPos('Manual', windowName)
    L_inv = cv2.getTrackbarPos('L_inv', windowName)
    L_th = cv2.getTrackbarPos('L_th', windowName)
    a_inv = cv2.getTrackbarPos('a_inv', windowName)
    a_th = cv2.getTrackbarPos('a_th', windowName)
    b_inv = cv2.getTrackbarPos('b_inv', windowName)
    b_th = cv2.getTrackbarPos('b_th', windowName)

    if manual_switch == 1:
        if L_inv == 1:
            _, threshed_L = cv2.threshold(l, L_th, 255, cv2.THRESH_BINARY_INV)
        else:
            _, threshed_L = cv2.threshold(l, L_th, 255, cv2.THRESH_BINARY)

        if a_inv == 1:
            _, threshed_a = cv2.threshold(a, a_th, 255, cv2.THRESH_BINARY_INV)
        else:
            _, threshed_a = cv2.threshold(a, a_th, 255, cv2.THRESH_BINARY)

        if b_inv == 1:
            _, threshed_b = cv2.threshold(b, b_th, 255, cv2.THRESH_BINARY_INV)
        else:
            _, threshed_b = cv2.threshold(b, b_th, 255, cv2.THRESH_BINARY)
    else:
        if L_inv == 1:
            _, threshed_L = cv2.threshold(l, L_th, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, threshed_L = cv2.threshold(l, L_th, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if a_inv == 1:
            _, threshed_a = cv2.threshold(a, a_th, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, threshed_a = cv2.threshold(a, a_th, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if b_inv == 1:
            _, threshed_b = cv2.threshold(b, b_th, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, threshed_b = cv2.threshold(b, b_th, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    masked_L = cv2.bitwise_and(temp, temp, mask=threshed_L)
    masked_a = cv2.bitwise_and(temp, temp, mask=threshed_a)
    masked_b = cv2.bitwise_and(temp, temp, mask=threshed_b)
    masked_all = cv2.bitwise_and(masked_L, masked_a, mask=threshed_b)
    cv2.putText(masked_b, 'masked_b', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(masked_L, 'masked_L', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(masked_a, 'masked_a', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    stacked1 = np.vstack((masked_L, masked_a))
    stacked2 = np.vstack((masked_b, masked_all))
    stacked = np.hstack((stacked1, stacked2))

    cv2.imshow(windowName, cv2.resize(stacked, (0, 0), fx=0.5, fy=0.5))
