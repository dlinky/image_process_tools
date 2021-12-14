import cv2
import os
import numpy as np

file_list = [_ for _ in os.listdir() if _.endswith('.jpg')]
img = cv2.imread(file_list[0])
l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
l = cv2.cvtColor(l, cv2.COLOR_GRAY2BGR)
a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

cv2.putText(img, 'BGR', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(l, 'L', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(a, 'a', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(b, 'b', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

stack1 = np.hstack((img, l))
stack2 = np.hstack((a, b))
stack = np.vstack((stack1, stack2))

cv2.imshow('colorspace', stack)
cv2.waitKey(0)