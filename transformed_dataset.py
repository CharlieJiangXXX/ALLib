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
        def transform_single(value):
            value = list(value)
            if type(value[0]) is not torch.Tensor:
                value[0] = self.transformer(value[0])
            return tuple(value)

        if isinstance(idx, list):
            values = []
            for i in idx:
                value = self.dataset[self.indices[i]]
                values.append(transform_single(value))
            return values

        return transform_single(self.dataset[self.indices[idx]])

    def __len__(self):
        return len(self.indices)
