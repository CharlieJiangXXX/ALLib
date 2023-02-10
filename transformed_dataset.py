from torch.utils import data
import torch


class TransformedDataset(data.Dataset):
    """
    Transforms a dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
        transformer (LambdaType): (idx, sample) -> transformed_sample
    """

    def __init__(self, dataset, *, transformer=None, vision_transformer=None):
        self.dataset = dataset
        self.indices = torch.arange(len(dataset))
        assert not transformer or not vision_transformer
        if transformer:
            self.transformer = transformer
        else:
            self.transformer = lambda _, data_label: (vision_transformer(data_label[0]), data_label[1])

    def __getitem__(self, idx):
        if isinstance(idx, list):
            values = []
            for i in idx:
                value = self.dataset[self.indices[i]]
                values.append((self.transformer(value[0]), value[1]))
            return values

        x = self.transformer(self.dataset[self.indices[idx]][0])
        return x, self.dataset[self.indices[idx]][1]

    def __len__(self):
        return len(self.indices)
