from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import os
from functools import cmp_to_key


class MedImageFolders(Dataset):
    def __init__(self, folders: list[str], tile_num: int = 500, transform=None, target_transform=None):
        self.folders = folders
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        self.make_dataset(tile_num)
        self.classes = ["Normal", "HCC"]

    def make_dataset(self, tile_num: int):
        def get_stats_from_name(name: str) -> (int, int, int, int):
            h = int(name[-21:-17])
            s = int(name[-16:-12])
            v = int(name[-11:-7])
            pix_ratio = int(name[-6:-4])
            return h, s, v, pix_ratio

        def better(tile1: str, tile2: str) -> int:
            tile1_stats = get_stats_from_name(tile1)
            tile2_stats = get_stats_from_name(tile2)
            if tile1_stats[1] > tile2_stats[1]:
                return 1
            if tile1_stats[1] == tile2_stats[1]:
                return 0
            return -1

        for folder in self.folders:
            files = os.listdir(folder)

            # SLIDENAME_slice_${i}_${j}_${h}_${s}_${v}_${pix_ratio}.jpg
            # h, s, v: 4 digits; pix_ratio: 2 digits
            files = sorted(files, key=cmp_to_key(better), reverse=True)
            class_idx = int("Rat_HCC_HE" in folder)

            top_tiles = [(os.path.join(folder, elem), class_idx) for elem in files[:tile_num]]

            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1 import ImageGrid
            from matplotlib.image import imread

            def temp_plot(folder, top_tiles):
                images = []
                for img in top_tiles:
                    image = imread(img[0])
                    images.append(image)

                fig = plt.figure(figsize=(10., 10.))
                grid = ImageGrid(fig, 111, nrows_ncols=(7, 7), axes_pad=0.1)

                for ax, im in zip(grid, images):
                    ax.imshow(im)

                fig.suptitle(folder)
                plt.show()

            #temp_plot(folder, top_tiles)

            self.samples.extend(top_tiles)

    def __getitem__(self, idx):
        def get_single(idx):
            path, label = self.samples[idx]
            image = default_loader(path)
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)

            return image, label, path

        if isinstance(idx, list):
            values = []
            for i in idx:
                values.append(get_single(i))
            return values

        return get_single(idx)

    def __len__(self):
        return len(self.samples)



