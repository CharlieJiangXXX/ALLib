from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import os


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

        def better(tile1: str, tile2: str) -> bool:
            tile1_stats = get_stats_from_name(tile1)
            tile2_stats = get_stats_from_name(tile2)
            if tile1_stats[1] > tile2_stats[1]:
                return True
            return False

        for folder in self.folders:
            top_tiles = []

            files = os.listdir(folder)
            class_idx = int("Rat_HCC_HE" in folder)

            # SLIDENAME_slice_${i}_${j}_${h}_${s}_${v}_${pix_ratio}.jpg
            # h, s, v: 4 digits; pix_ratio: 2 digits

            for name in files:
                full_name = os.path.join(folder, name)
                if not top_tiles:
                    top_tiles.append((full_name, class_idx))
                    continue

                if better(full_name, top_tiles[-1][0]):
                    top_tiles[-1:-1] = [(full_name, class_idx)]  # Append before last
                else:
                    top_tiles.append((full_name, class_idx))
                if len(top_tiles) > tile_num:
                    top_tiles.pop()

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
