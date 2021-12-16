import copy

import cv2
import os

import numpy as np

file = [_ for _ in os.listdir() if _.endswith('.jpg')]
img = cv2.imread(file[0])


lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
b, g, r = cv2.split(img)
l, a, b = cv2.split(lab)
h, s, v = cv2.split(hsv)
tozero = cv2.THRESH_TOZERO
inv = cv2.THRESH_TOZERO_INV
otsu = cv2. THRESH_OTSU

labels = ['R', 'G', 'B', 'L', 'a', 'b', 'H', 'S', 'V']
cvt_imgs = [r, g, b,
             l, a, b,
             h, s, v]
th_methods = [inv, inv, inv,
             inv, tozero, inv,
             tozero, tozero, inv]
th_imgs = []
th_rets = []

cv2.imshow('a', a)
cv2.waitKey(0)

for th_method, cvt_img, label in zip(th_methods, cvt_imgs, labels):
    ret, th_img = cv2.threshold(cvt_img, 0, 255, th_method + otsu)
    th_img = cv2.cvtColor(th_img, cv2.COLOR_GRAY2BGR)
    th_rets.append(ret)
    th_imgs.append(th_img)
    #cv2.putText(th_img, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 20, cv2.LINE_AA)
    #cv2.putText(th_img, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)

stack1 = np.hstack((th_imgs[0], th_imgs[1], th_imgs[2]))
stack2 = np.hstack((th_imgs[3], th_imgs[4], th_imgs[5]))
stack3 = np.hstack((th_imgs[6], th_imgs[7], th_imgs[8]))
stacked = np.vstack((stack1, stack2, stack3))

cv2.imshow('win', cv2.resize(stacked, (1000, 1000)))
cv2.waitKey(0)

th_s = cv2.cvtColor(th_imgs[7], cv2.COLOR_BGR2GRAY)
th_s[th_s == 0] = th_rets[7]
value_max = np.max(th_s)
_, reth_s = cv2.threshold(th_s, int((value_max + th_rets[7])/2), 255, cv2.THRESH_BINARY)
cv2.imshow('reth', reth_s)
cv2.waitKey(0)