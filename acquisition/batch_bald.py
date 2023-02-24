import torch.nn as nn
from torch.utils.data import random_split, Dataset

from acquisition.bald import *


class ATBatchDisagreement(ATDisagreement):
    def __init__(self, batch_size: int, model: nn.Module):
        super().__init__(batch_size, model)
        self._numMCSamples = 10000  # Number of MC samples for label combinations
        self._numSubPoolDataPoints = 500  # number of datapoints in the sub-pool from which we acquire

    def select_batch(self, pool_data: Dataset, num_features: int = 10, k: int = 100) -> np.array:
        num_extra = len(pool_data) - self._numSubPoolDataPoints

        # performing BatchBALD on the whole pool is very expensive, so we take
        # a random subset of the pool. Even if we don't have enough data left to
        # split, we still need to call random_split to avoid messing up the indexing
        # later on.
        sub_pool_data, _ = random_split(pool_data, [self._numSubPoolDataPoints, num_extra] if num_extra > 0 else [len(pool_data), 0])

        # Forward pass on the pool once to get class probabilities for each x
        with torch.no_grad():
            pool_loader = DataLoader(sub_pool_data, batch_size=self._subBatchSize, pin_memory=True,
                                     shuffle=False)
            pool_p_y = torch.zeros(len(sub_pool_data), num_features, k)
            for index, sample in enumerate(pool_loader):
                sample = sample[0].to(self._device)
                pool_p_y[index:index + sample.shape[0]] = torch.stack([self._model(sample) for _ in range(k)],
                                                                      dim=1).permute(0, 2, 1)

        # this only need to be calculated once so we pull it out of the loop
        entropy2 = (element_entropy(pool_p_y).sum(dim=(1, 2)) / k).to(self._device)

        # get all class combinations
        c_1_to_n = class_combinations(num_features, self._batchSize, self._numMCSamples)

        # tensor of size [m x k]
        p_y_1_to_n_minus_1 = None

        # store the indices of the chosen datapoints in the subpool
        best_sub_local_indices = []
        # create a mask to keep track of which indices we've chosen
        remaining_indices = torch.ones(len(sub_pool_data), dtype=torch.bool).to(self._device)
        for n in range(self._batchSize):
            # tensor of size [N x m x l]
            p_y_n = pool_p_y[:, c_1_to_n[:, n], :].to(self._device)
            # tensor of size [N x m x k]
            p_y_1_to_n = torch.einsum('mk,pmk->pmk', p_y_1_to_n_minus_1, p_y_n) \
                if p_y_1_to_n_minus_1 is not None else p_y_n

            # and compute the left entropy term
            entropy1 = element_entropy(p_y_1_to_n.mean(axis=2)).sum(dim=1)
            # scores is a vector of scores for each element in the pool.
            # mask by the remaining indices and find the highest scoring element
            scores = entropy1 - entropy2
            # print(scores)
            best_local_index = torch.argmax(scores - np.inf * (~remaining_indices)).item()
            # print(f'Best idx {best_local_index}')
            best_sub_local_indices.append(best_local_index)
            # save the computation for the next batch
            p_y_1_to_n_minus_1 = p_y_1_to_n[best_local_index]
            # remove the chosen element from the remaining indices mask
            remaining_indices[best_local_index] = False

        # we've subset-ed our dataset twice, so we need to go back through
        # subset indices twice to recover the global indices of the chosen data
        best_local_indices = np.array(sub_pool_data.indices)[best_sub_local_indices]
        best_global_indices = np.array(pool_data.indices)[best_local_indices]
        return best_global_indices
