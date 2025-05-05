from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Normalization(ABC):
    @abstractmethod
    def penalty(self, weights: NDArray) -> float:
        pass

    @abstractmethod
    def grad(self, weights: NDArray) -> NDArray:
        pass


class L1(Normalization):
    def penalty(self, weights: NDArray) -> float:
        return np.sum(np.abs(weights))

    def grad(self, weights: NDArray) -> NDArray:
        return np.sign(weights)


class L2(Normalization):
    def penalty(self, weights: NDArray) -> float:
        return 0.5 * np.sum(weights**2)

    def grad(self, weights: NDArray) -> NDArray:
        return weights


class ElasticNet(Normalization):
    def __init__(self, ratio: float = 0.5, lambda_: float = 1e-4) -> None:
        self.ratio = ratio
        self.lambda_ = lambda_

    def penalty(self, weights: NDArray) -> float:
        return self.ratio * L1().penalty(weights) + (1 - self.ratio) * L2().penalty(
            weights
        )

    def grad(self, weights: NDArray) -> NDArray:
        return self.ratio * L1().grad(weights) + (1 - self.ratio) * L2().grad(weights)
