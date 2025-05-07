from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Optimizer(ABC):
    def __init__(self, params: list[tuple[NDArray, NDArray]], lr: float = 1e-2) -> None:
        self.params = params
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        pass


class SGD(Optimizer):
    def step(self) -> None:
        for param, grad in self.params:
            param -= self.lr * grad


class Adagrad(Optimizer):
    def __init__(
        self,
        params: list[tuple[NDArray, NDArray]],
        lr: float = 1e-2,
        eps: float = 1e-15,
    ) -> None:
        super().__init__(params, lr)

        self.eps = eps
        self.cache: list[NDArray] = [np.zeros_like(param) for param, _ in params]

    def step(self) -> None:
        for idx, (p, grad) in enumerate(self.params):
            self.cache[idx] += grad**2
            lr = self.lr / (np.sqrt(self.cache[idx]) + self.eps)
            p -= lr * grad


class RMSProp(Optimizer):
    def __init__(
        self,
        params: list[tuple[NDArray, NDArray]],
        lr: float = 1e-2,
        beta: float = 0.9,
        eps: float = 1e-15,
    ) -> None:
        super().__init__(params, lr)

        self.beta, self.eps = beta, eps
        self.cache: list[NDArray] = [np.zeros_like(param) for param, _ in params]

    def step(self) -> None:
        for idx, (p, grad) in enumerate(self.params):
            self.cache[idx] = self.beta * self.cache[idx] + (1 - self.beta) * grad**2
            lr = self.lr / (np.sqrt(self.cache[idx]) + self.eps)
            p -= lr * grad


class Adam(Optimizer):
    def __init__(
        self,
        params: list[tuple[NDArray, NDArray]],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)

        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for idx, (p, grad) in enumerate(self.params):
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad**2)

            m_hat = self.m[idx] / (1 - self.beta1**self.t)
            v_hat = self.v[idx] / (1 - self.beta2**self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
