from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Module(ABC):
    def __init__(self):
        self._params: list[NDArray] = []
        self._grads: list[NDArray] = []

    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def backward(self, grad: NDArray) -> NDArray:
        pass

    def parameters(self) -> list[tuple[NDArray, NDArray]]:
        return list(zip(self._params, self._grads))

    def zero_grad(self) -> None:
        for grad in self._grads:
            grad.fill(0)


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: NDArray) -> NDArray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: NDArray) -> NDArray:
        grad_out = grad
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()


class DropOut(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
        self.p = p
        self.mask: NDArray = np.array([])

    def forward(self, x: NDArray) -> NDArray:
        # if self.training and self.p > 0:
        self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
        return x * self.mask
        # return x

    def backward(self, grad: NDArray) -> NDArray:
        # if self.training and self.p > 0:
        return grad * self.mask
        # return grad


# Linear layer: y = Wx + b
class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        weight: NDArray = np.random.randn(out_features, in_features) * np.sqrt(
            2 / in_features
        )
        bias: NDArray = np.zeros((out_features, 1))

        self._params = [weight, bias]
        self._grads = [np.zeros_like(weight), np.zeros_like(bias)]

        self._input: NDArray = np.zeros((in_features, 1))

    def forward(self, x: NDArray) -> NDArray:
        self._input = x
        W, b = self._params
        return W @ x + b

    # grad it's dLoss/dy
    def backward(self, grad: NDArray) -> NDArray:
        W, b = self._params
        dW, db = self._grads
        x = self._input

        dW[...] = grad @ x.T
        db[...] = np.sum(grad, axis=1, keepdims=True)

        return W.T @ grad


# Activation: ReLU
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self._cache: NDArray = np.zeros_like(1)

    def forward(self, x: NDArray) -> NDArray:
        self._cache = x
        return np.maximum(0, x)

    def backward(self, grad: NDArray) -> NDArray:
        return grad * (self._cache > 0)


# Activation: Sigmoid
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self._cache: NDArray = np.zeros_like(1)

    def forward(self, x: NDArray) -> NDArray:
        self._cache = 1 / (1 + np.exp(-x))
        return self._cache

    def backward(self, grad: NDArray) -> NDArray:
        return self._cache * (1 - self._cache)
