from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import os


class MedImageFolders(Dataset):
    def __init__(self, folders: list[str], transform=None, target_transform=None):
        self.folders = folders
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        self.make_dataset()
        self.classes = ["Normal", "Rat_HCC_HE"]

    def make_dataset(self):
        for folder in self.folders:
            files = os.listdir(folder)
            class_idx = int("Rat_HCC_HE" in folder)
            for name in files:
                self.samples.append((os.path.join(folder, name), class_idx))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = default_loader(path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.samples)
