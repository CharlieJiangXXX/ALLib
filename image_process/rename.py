# Renaming original tiles that didn't contain HSV information in their names

import os
from PIL import Image
import numpy as np
from skimage.color import rgb2hsv, rgb2gray
from pathlib import Path

ROOT_DIR = "F:\\Processed"
folders = []
for x in os.walk(ROOT_DIR):
    if x[0].count("\\") > 2:
        folders.append(x[0])

for folder in folders:
    files = os.listdir(folder)
    for name in files:
        full_name = os.path.join(folder, name)
        img = Image.open(full_name)
        pix = np.array(img)
        hsv_img = rgb2hsv(pix[:, :, :3])
        h = np.mean(hsv_img[:, :, 0])
        s = np.mean(hsv_img[:, :, 1])
        v = np.mean(hsv_img[:, :, 2])
        gray_img = rgb2gray(pix)
        binary = gray_img > 0.8
        pix_ratio = np.count_nonzero(binary) / (binary.shape[0] * binary.shape[1])
        info_str = '{:.4f}_{:.4f}_{:.4f}_{:02d}'.format(h, s, v, int(pix_ratio * 100))
        p = Path(full_name)
        new_name = "{0}_{2}{1}".format(Path.joinpath(p.parent, p.stem), p.suffix, info_str.replace('0.', ''))
        print("Previous name: {}".format(full_name))
        print("New name: {}".format(new_name))

        # Some brute force magic
        if full_name[-11:-7].isnumeric():
            print("Filename is already formatted!")
        else:
            os.rename(full_name, new_name)
            print("Renamed.")
        print()