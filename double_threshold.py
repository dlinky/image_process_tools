import os
import numpy as np
import cv2



file = [_ for _ in os.listdir() if _.endswith('.jpg')][0]
img = cv2.imread(file)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
a = cv2.split(lab)[1]
ret, threshed1 = cv2.threshold(a, 0, 255, cv2.THRESH_TRIANGLE + cv2.THRESH_TOZERO)
threshed2 = threshed1.copy()
threshed2[threshed1 == 0] = ret+1
_, threshed3 = cv2.threshold(threshed2, 0, 255, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY)

stack1 = np.hstack((a, threshed1))
stack2 = np.hstack((threshed2, threshed3))
stack = np.vstack((stack1, stack2))

cv2.imshow('thersh', stack)
cv2.waitKey(0)