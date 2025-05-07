import numpy as np
from numpy.typing import NDArray


class DataLoader:
    def __init__(
        self, x: NDArray, y: NDArray, batch_size: int = 32, shuffle: bool = True
    ):
        assert x.shape[0] == y.shape[0], "The number of examples in x and y must match"

        y = y[:, np.newaxis] if y.ndim == 1 else y
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = x.shape[0]
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start in range(0, self.num_samples, self.batch_size):
            end = start + self.batch_size
            batch_idx = self.indices[start:end]

            yield self.x[batch_idx], self.y[batch_idx]

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
