import os

import cv2
import numpy as np


def watershed(img, dist_threshold):
    # binaray image로 변환
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    ret, thresh = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        flood_filled = cv2.drawContours(thresh, [c], 0, (255, 255, 255), -1)

    # Morphology의 opening, closing을 통해서 노이즈나 Hole제거
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(flood_filled, cv2.MORPH_OPEN, kernel, iterations=2)

    # dilate를 통해서 확실한 Backgroud
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # distance transform을 적용하면 중심으로 부터 Skeleton Image를 얻을 수 있음.
    # 즉, 중심으로 부터 점점 옅어져 가는 영상.
    # 그 결과에 thresh를 이용하여 확실한 FG를 파악
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, dist_threshold / 100 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Background에서 Foregrand를 제외한 영역을 Unknow영역으로 파악
    unknown = cv2.subtract(sure_bg, sure_fg)

    # FG에 Labelling작업
    _, markers, stats, _ = cv2.connectedComponentsWithStats(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # watershed를 적용하고 경계 영역, 배경 삭제
    markers = cv2.watershed(img, markers)
    markers[markers == -1] = 0
    markers[markers == 1] = 0

    '''for index, cell in enumerate(stats):
        print(cell)
        cv2.putText(img, str(index), (cell[0], cell[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    img[markers == -1] = [255, 0, 0]

    images = [a, thresh, sure_bg, dist_transform, sure_fg, unknown, markers, img]
    titles = ['Gray', 'Binary', 'Sure BG', 'Distance', 'Sure FG', 'Unknow', 'Markers', 'Result']

    for i in range(len(images)):
        plt.subplot(2, 4, i + 1), plt.imshow(images[i]), plt.title(titles[i]), plt.xticks([]), plt.yticks([])

    plt.show()'''

    return markers


def get_best_threshold(img, th_range):
    nums_cells = []
    for dist_threshold in np.arange(th_range[0], th_range[1]):
        markers = watershed(img, dist_threshold)
        for count in range(2, 300):
            if len(markers[markers == count]) == 0:
                nums_cells.append(count)
                break
        print('th = %d, count = %d' % (dist_threshold, nums_cells[-1]))
    best_threshold = th_range[0] + nums_cells.index(max(nums_cells))
    return best_threshold


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


def main():
    file =[_ for _ in os.listdir() if _.endswith('.jpg')][0]
    img = cv2.imread(file)
    dist_threshold = get_best_threshold(img, (10, 90))
    markers = watershed(img, dist_threshold)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(markers))
    label_img = cvt_label_to_image(labels)
    cv2.imshow('win', label_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()