import os
import openslide
import numpy as np

from PIL import Image
from skimage.color import rgb2hsv, rgb2gray
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SlideObject:
    name: str
    slide: openslide.OpenSlide


def get_images(base_dir):
    mr_images = []
    for file in os.listdir(base_dir):
        name = os.path.join(base_dir, file)
        if os.path.isdir(name):
            mr_images.extend(get_images(name))
        elif os.path.splitext(name)[-1].lower() == '.ndpi':
            try:
                mr_images.append(SlideObject(name=name, slide=openslide.OpenSlide(name)))
            except:
                continue
    return mr_images


print("Getting slides...")
mr_images = get_images("/media/cjiang/Extreme SSD/Rat_HCC_HE")
for img in mr_images:
    print(img.name)
print("Slides obtained.")
x_stride = y_stride = 1000
x_patch = y_patch = 2000
slice_x = slice_y = 1000
level = 0

for obj in mr_images:
    print(f"Processing {obj.name}...")
    for i in range(obj.slide.level_dimensions[level][0] // x_stride):  # round up?
        for j in range(obj.slide.level_dimensions[level][1] // y_stride):
            image_patch = obj.slide.read_region((x_stride * i, y_stride * j), level, (x_patch, y_patch))
            pix = np.array(image_patch)
            hsv_img = rgb2hsv(pix[:, :, :3])
            h = np.mean(hsv_img[:, :, 0])
            s = np.mean(hsv_img[:, :, 1])
            v = np.mean(hsv_img[:, :, 2])
            gray_img = rgb2gray(pix)
            binary = gray_img > 0.8
            pix_ratio = np.count_nonzero(binary) / (binary.shape[0] * binary.shape[1])
            print(f"Processed segment {i}_{j}.")

            if s > 0.05 and h > 0.6 and v > 0.5 and pix_ratio < 0.95:
                print("Saving...")
                info_str = 'slice_{}_{}_{:.4f}_{:.4f}_{:.4f}_{:02d}'.format(i, j, h, s, v, int(pix_ratio * 100))
                p = Path(obj.name)
                new_path = "{}_{}.jpg".format(Path.joinpath(p.parent, p.stem), info_str.replace('0.', ''))
                print("Previous name: {}".format(obj.name))
                print("New name: {}".format(new_path))

                image_patch = image_patch.resize((slice_x, slice_y))
                image_arr = np.array(image_patch)
                img = Image.fromarray(image_arr).convert("RGB")
                if not os.path.exists(new_path):
                    img.save(new_path)
                else:
                    print("File already exists. Skipping...")
                print()
