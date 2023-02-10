#!/usr/bin/python
from acquisition.acquirer import *


# @class ATDisagreement
# @abstract BALD acquisition function.

class ATDisagreement(ATAcquirer):
    def score(self, datapoints: torch.Tensor, k: int = 100) -> np.array:
        # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

        with torch.no_grad():
            # take k monte-carlo samples of forward pass without dropout
            y = torch.stack([self._model(datapoints.to(self._device)) for i in range(k)], dim=1)
            entropy_1 = element_entropy(y.mean(axis=1)).sum(dim=1)
            entropy_2 = element_entropy(y).sum(dim=(1, 2)) / k

            return entropy_1 - entropy_2
