import os
import cv2


def empty(self):
    pass


img = cv2.imread('101-1.jpg')


windowName = 'threshold'
cv2.namedWindow(windowName)
cv2.createTrackbar('Otsu', windowName, 0, 1, empty)
cv2.createTrackbar('threshold', windowName, 0, 255, empty)

while True:
    if cv2.waitKey(10) == 27:
        break
    switch = cv2.getTrackbarPos('Otsu', windowName)
    if switch == 1:
        thMethod = cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU
    else:
        thMethod = cv2.THRESH_TOZERO_INV
    thValue = cv2.getTrackbarPos('threshold', windowName)
    temp = img.copy()

    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    ret, threshed = cv2.threshold(gray, thValue, 255, thMethod)
    threshed = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

    if switch == 1:
        text = ret
    else:
        text = thValue
    cv2.putText(threshed, str(text), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
    cv2.imshow(windowName, threshed)