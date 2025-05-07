import numpy as np
from numpy.typing import DTypeLike, NDArray


class CrossEntropy:
    def __init__(self):
        self.probs: NDArray = np.array([])

    def forward(self, logits: NDArray, labels: NDArray) -> float:
        max_logit: DTypeLike = np.max(logits, axis=1, keepdims=True)  # Avoid overflow
        exps = np.exp(logits - max_logit)
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)  # Softmax(logits)

        m = labels.shape[0]
        loss = -np.sum(labels * np.log(self.probs + 1e-15)) / m

        return loss

    def backward(self, labels: NDArray) -> NDArray:
        m = labels.shape[0]
        return (self.probs - labels) / m
