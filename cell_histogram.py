import cv2
import numpy as np
import matplotlib.pyplot as plt


def cluster_boxes(num_clst, data, bins):
    data = np.array(data.astype('int16').T)


    mxmn = [np.min(data), np.max(data)]
    quan = (mxmn[1] - mxmn[0]) / num_clst / 2
    clst_mu = np.array([quan * (2 * i + 1) + mxmn[0] for i in range(num_clst)])

    diff = np.abs(data - clst_mu.reshape((num_clst, 1))).T
    clst = np.where(diff == diff.min(axis=1).reshape((len(data), 1)))[1]
    if len(clst) > len(data):
        for count in range(len(clst) - len(data)):
            clst = np.delete(clst, len(clst) - 1)

    for i in range(num_clst):
        clst_mu[i] = np.mean(data[np.where(clst == i)[0]])

    maxEpoch = 5
    for epoch in range(maxEpoch):
        print(f'epoch #{epoch} : ')
        diff = np.abs(data - clst_mu.reshape((num_clst, 1))).T
        clst = np.where(diff == diff.min(axis=1).reshape((len(data), 1)))[1]
        for i in range(num_clst):
            print(f'class {i}')
            clst_mu[i] = np.mean(data[np.where(clst == i)[0]])
            clst_min = np.min(data[np.where(clst == i)[0]])
            clst_max = np.max(data[np.where(clst == i)[0]])
            print(f'min = {clst_min}, max = {clst_max}, mu = {clst_mu[i]}')


img = cv2.imread('101-1.jpg')
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

ret, threshed = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO)

bins = np.arange(0, 256)
hist = cv2.calcHist([threshed], [0], threshed, [256], [0, 256])
cluster_boxes(2, hist, bins)