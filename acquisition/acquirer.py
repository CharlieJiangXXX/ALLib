#!/usr/bin/python
from itertools import combinations_with_replacement

import numpy as np
import torch
from torch.utils.data import DataLoader


# @function convert_to_numpy
# @abstract Convert tensor to numpy.
# @param t The tensor to convert.
# @result A numpy ndarray.

def convert_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().numpy()


# @function class_combinations
# @discussion Generates an array of n-element combinations where each element is one of
#             the c classes (an integer). If m is provided and m < n^c, then instead of all
#             n^c combinations, m combinations are randomly sampled.
# @param c The number of classes
# @param n The number of elements in each combination
# @param m The number of desired combinations (default: {np.inf})
# @result An [m x n] or [n^c x n] array of integers in [0, c)

def class_combinations(c: int, n: int, m: int = np.inf) -> np.ndarray:
    if m < c ** n:
        # randomly sample combinations
        return np.random.randint(c, size=(int(m), n))
    else:
        p_c = combinations_with_replacement(np.arange(c), n)
        return np.array(list(iter(p_c)), dtype=int)


# @function element_entropy
# @abstract Compute the element-wise entropy of x
# @discussion x An array of probabilities in (0,1)
# @discussion eps Prevent failure on x == 0
# @result The entropy of element x.

def element_entropy(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = x + eps
    return -num * torch.log(num)


def nan_in_tensor(x):
    return torch.isnan(x).any()


def remove_occurrences_from_list(l, items):
    return list(np.setdiff1d(np.array(l, dtype=int),
                             np.array(items, dtype=int), assume_unique=True))


def move_data(indices, from_subset, to_subset):
    from_subset.indices = remove_occurrences_from_list(from_subset.indices, indices)
    if isinstance(to_subset.indices, list):
        to_subset.indices.extend(indices)
    elif isinstance(to_subset.indices, np.ndarray):
        to_subset.indices = np.concatenate([to_subset.indices, np.array(indices)])


# @class ATAcquirer
# @abstract Base class for all acquisition functions.

class ATAcquirer:
    def __init__(self, batch_size, model):
        self._batchSize = batch_size
        self._subBatchSize = 128
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = model

    # @function score
    # @abstract Paralleled acquisition scoring function.
    # @discussion       This function must be implemented by all subclasses based on the nature of the
    #                   algorithms they represent. Based on the current model, it evaluates @datapoints
    #                   and yields a tensor of acquisition scores.
    # @param datapoints The datapoints to evaluate.
    # @result           A vector of acquisition scores.

    def score(self, datapoints: torch.Tensor) -> np.array:
        return torch.zeros(len(datapoints))

    # @function select_batch
    # @abstract Score every datapoint in the pool under the model.
    # @param pool_data The data pool whose datapoints the function analyze.
    # @result          The best local indices.

    def select_batch(self, pool_data) -> np.array:
        pool_loader = DataLoader(pool_data, batch_size=self._subBatchSize,
                                 pin_memory=True, shuffle=False)
        scores = torch.zeros(len(pool_data)).to(self._device)
        for index, sample in enumerate(pool_loader):
            scores[index:index + sample.shape[0]] = self.score(sample.to(self._device))

        best_local_indices = torch.argsort(scores)[-self._batchSize:]
        return np.array(pool_data.indices)[convert_to_numpy(best_local_indices)]
