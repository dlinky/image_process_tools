import cv2
import numpy as np


def cvt_label_to_image(labels):
    """
    레이블링된 전처리 이미지를 시각화
    0은 검은색, 나머지는 Hue값 돌아가면서 사용
    """
    image = np.zeros((len(labels), len(labels[0]), 3), 'uint8')
    for r, line in enumerate(labels):
        for c, item in enumerate(line):
            if item == 0:
                image[r][c] = [0, 0, 0]
            else:
                image[r][c] = [(item * 5) % 180, 255, 255]
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def watershed(opening):
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 / 100 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg, connectivity=4)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    markers[markers == -1] = 0
    markers[markers == 1] = 0

    _, just_labels = cv2.connectedComponents(opening, connectivity=4)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(markers), connectivity=4)

    return just_labels, labels, len(stats)


img = cv2.imread('cropped.jpg')

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
ret, thresh = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    flood_filled = cv2.drawContours(thresh, [c], 0, (255, 255, 255), -1)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(flood_filled, cv2.MORPH_OPEN, kernel, iterations=2)

th_list = np.arange(20, 80, 0.01)
counts = np.zeros_like(th_list)
for idx, th in enumerate(th_list):
    _, _, counts[idx] = watershed(opening)

th = th_list[np.argmax(counts)]

just_labels, labels, count = watershed(opening)

just_label_img = cvt_label_to_image(just_labels)
label_img = cvt_label_to_image(labels)
stack = np.hstack((just_label_img, label_img))
cv2.imshow('win', stack)
cv2.waitKey(0)
