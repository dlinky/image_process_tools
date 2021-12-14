import cv2
import numpy as np
import os
import copy

file_list = [_ for _ in os.listdir() if _.endswith('.jpg')]
img = cv2.imread(file_list[0])
l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

ret_cell, threshed_a = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO)
masked_a = copy.deepcopy(threshed_a)
masked_a[masked_a == 0] = ret_cell
ret_nucleus, rethreshed_a = cv2.threshold(masked_a, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
print(ret_cell, ret_nucleus)
l = cv2.cvtColor(l, cv2.COLOR_GRAY2BGR)
a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

stack1 = np.hstack((img, cv2.cvtColor(threshed_a, cv2.COLOR_GRAY2BGR)))
stack2 = np.hstack((cv2.cvtColor(masked_a, cv2.COLOR_GRAY2BGR), cv2.cvtColor(rethreshed_a, cv2.COLOR_GRAY2BGR)))
stack = np.vstack((stack1, stack2))
cv2.imshow('win', stack)
cv2.waitKey(0)