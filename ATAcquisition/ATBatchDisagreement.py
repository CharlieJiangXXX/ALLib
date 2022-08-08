from torch.utils import data

from ATDisagreement import *


class ATBatchDisagreement(ATDisagreement):
    def __init__(self, batch_size, device, model, pool_data):
        super().__init__(batch_size, device, model)
        self.m = 1e4  # number of MC samples for label combinations
        self.num_sub_pool = 500  # number of datapoints in the sub-pool from which we acquire
        self._pool_data = pool_data

    def batch_score(self, k: int = 100) -> np.array:
        c = 10  # number of classes

        # performing BatchBALD on the whole pool is very expensive, so we take
        # a random subset of the pool.
        num_extra = len(self._pool_data) - self.num_sub_pool
        if num_extra > 0:
            sub_pool_data, _ = torch.utils.data.random_split(self._pool_data, [self.num_sub_pool, num_extra])
        else:
            # even if we don't have enough data left to split, we still need to
            # call random_splot to avoid messing up the indexing later on
            sub_pool_data, _ = torch.utils.data.random_split(self._pool_data, [len(self._pool_data), 0])

        # Forward pass on the pool once to get class probabilities for each x
        with torch.no_grad():
            pool_loader = torch.utils.data.DataLoader(sub_pool_data,
                                                      batch_size=self.processing_batch_size, pin_memory=True,
                                                      shuffle=False)
            pool_p_y = torch.zeros(len(sub_pool_data), c, k)
            for batch_idx, (data, _) in enumerate(pool_loader):
                end_idx = batch_idx + data.shape[0]
                pool_p_y[batch_idx:end_idx] = torch.stack([self._model(data.to(self.device)) for i in range(k)],
                                                          dim=1).permute(0, 2, 1)

        # this only need to be calculated once so we pull it out of the loop
        H2 = (H(pool_p_y).sum(axis=(1, 2)) / k).to(self.device)

        # get all class combinations
        c_1_to_n = class_combinations(c, self.batch_size, self.m)

        # tensor of size [m x k]
        p_y_1_to_n_minus_1 = None

        # store the indices of the chosen datapoints in the subpool
        best_sub_local_indices = []
        # create a mask to keep track of which indices we've chosen
        remaining_indices = torch.ones(len(sub_pool_data), dtype=bool).to(self.device)
        for n in range(self.batch_size):
            # tensor of size [N x m x l]
            p_y_n = pool_p_y[:, c_1_to_n[:, n], :].to(self.device)
            # tensor of size [N x m x k]
            p_y_1_to_n = torch.einsum('mk,pmk->pmk', p_y_1_to_n_minus_1, p_y_n) \
                if p_y_1_to_n_minus_1 is not None else p_y_n

            # and compute the left entropy term
            H1 = H(p_y_1_to_n.mean(axis=2)).sum(axis=1)
            # scores is a vector of scores for each element in the pool.
            # mask by the remaining indices and find the highest scoring element
            scores = H1 - H2
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
