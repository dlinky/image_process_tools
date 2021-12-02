import os

import numpy as np
from matplotlib import pyplot as plt

import labelimg_xml

xml_dir = os.path.join(os.getcwd(), 'xml')
hist_dir = os.path.join(os.getcwd(), 'hist')
file_list = [_ for _ in os.listdir(xml_dir) if _.endswith('.xml')]

areas = [[] for _ in range(3)]
indexes = ['WBC', 'RBC', 'Platelets']
big_sizes = []
for file in file_list:
    title, table = labelimg_xml.read_xml(xml_dir, '/'+file)

    sizes = [(box[3] - box[1]) * (box[4] - box[2]) for box in table]
    big_sizes.extend(sizes)
    # hist, bins = np.histogram(sizes)
    hist = plt.hist(sizes, bins=20, range=(0, 10000))
    plt.savefig(os.path.join(hist_dir, file.replace('.xml', '.png')))
    plt.close()

hist = plt.hist(big_sizes, bins=20)
plt.savefig(os.path.join(hist_dir, 'all.png'))