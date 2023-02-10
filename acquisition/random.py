from acquirer import *


# @class ATRandom
# @abstract Random acquisition function

class ATRandom(ATAcquirer):
    def score(self, datapoints: torch.Tensor) -> np.array:
        return np.random.rand()
