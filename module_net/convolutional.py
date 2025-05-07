import numpy as np
from module import Module
from numpy.typing import NDArray


class Conv2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        fan_in = in_channels * kernel_size * kernel_size
        w = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / fan_in)
        b = np.zeros(out_channels)

        self._params = [w, b]
        self._grads = [np.zeros_like(w), np.zeros_like(b)]
        self._cache: tuple[NDArray, NDArray] = (np.zeros_like(w), np.zeros_like(w))

    def _out_size(self, size: int) -> int:
        return (size + 2 * self.padding - self.kernel_size) // self.stride + 1

    def forward(self, x: NDArray) -> NDArray:
        # x.shape: (batch_size, in_channels, height, width)
        # out.shape: (batch_size, out_channels, height_out, width_out)
        #
        # in_channels: 1 - Black/White, 3 - RGB
        # out_channels: Number of filters
        w, b = self._params
        stride, padding, kernel_size = self.stride, self.padding, self.kernel_size
        out_channels = w.shape[0]
        batch_size, in_channels, height, width = x.shape
        assert in_channels == w.shape[1]

        # pad
        paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
        x_pad = np.pad(x, paddings, mode="constant", constant_values=0)

        height_out = self._out_size(height)
        width_out = self._out_size(width)

        out = np.zeros((batch_size, out_channels, height_out, width_out))
        self._cache = (x, x_pad)

        for n in range(batch_size):
            for out_c in range(out_channels):
                kernel = w[out_c]
                bias = b[out_c]
                for i in range(height_out):
                    for j in range(width_out):
                        # h0 - height_stride, w0 - width_stride
                        h0, w0 = stride * i, stride * j
                        region = x_pad[
                            n, :, h0 : h0 + kernel_size, w0 : w0 + kernel_size
                        ]
                        out[n, out_c, i, j] = np.sum(region * kernel) + bias

        return out

    def backward(self, grad: NDArray) -> NDArray:
        # x.shape: (batch_size, in_channels, height, width)
        # grad.shape: (batch_size, out_channels, height_out, width_out)
        x, x_pad = self._cache
        w, b = self._params
        dw, db = self._grads

        stride, padding, kernel_size = self.stride, self.padding, self.kernel_size
        batch_size, in_channels, height, width = x.shape
        _, out_channels, height_out, width_out = grad.shape

        dw.fill(0)
        db.fill(0)

        dx_pad = np.zeros_like(x_pad)

        for n in range(batch_size):
            for out_c in range(out_channels):
                kernel = w[out_c]
                for i in range(height_out):
                    for j in range(width_out):
                        # h0 - height_stride, w0 - width_stride
                        h0, w0 = stride * i, stride * j

                        region = x_pad[
                            n, :, h0 : h0 + kernel_size, w0 : w0 + kernel_size
                        ]

                        grad_out = grad[n, out_c, i, j]

                        dw[out_c] += grad_out * region
                        db[out_c] += grad_out

                        dx_pad[n, :, h0 : h0 + kernel_size, w0 : w0 + kernel_size] += (
                            grad_out * kernel
                        )
        # unpad
        if padding > 0:
            return dx_pad[:, :, padding:-padding, padding:-padding]
        return dx_pad


class Flatten(Module):
    def __init__(self) -> None:
        super().__init__()
        self._cache: tuple[int, ...] = ()

    def forward(self, x: NDArray) -> NDArray:
        # x shape: (batch_size, ...)
        self._cache = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad: NDArray) -> NDArray:
        # grad shape: (batch_size, features)
        return grad.reshape(self._cache)


class MaxPool2D(Module):
    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self._cache: tuple[tuple[int, ...], NDArray] = (
            tuple(),
            np.zeros_like(1),
        )  # (x_shape, mask)

    def _out_size(self, size: int) -> int:
        return (size - self.kernel_size) // self.stride + 1

    def forward(self, x: NDArray) -> NDArray:
        # x shape: (batch_size, channels, height, width)
        # out.shape: (batch_size, channels, height_out, width_out)
        stride, kernel_size = self.stride, self.kernel_size
        batch_size, channels, height, width = x.shape
        height_out = self._out_size(height)
        width_out = self._out_size(width)

        out = np.zeros((batch_size, channels, height_out, width_out))
        mask = np.zeros_like(x, dtype=bool)

        for n in range(batch_size):
            for c in range(channels):
                for i in range(height_out):
                    for j in range(width_out):
                        # h0 - height_stride, w0 - width_stride
                        h0, w0 = i * stride, j * stride
                        window = x[n, c, h0 : h0 + kernel_size, w0 : w0 + kernel_size]
                        max_val = np.max(window)
                        out[n, c, i, j] = max_val
                        # create mask
                        mask[n, c, h0 : h0 + kernel_size, w0 : w0 + kernel_size] = (
                            window == max_val
                        )

        self._cache = (x.shape, mask)
        return out

    def backward(self, grad: NDArray) -> NDArray:
        # grad shape: (batch_size, channels, h_out, w_out)
        x_shape, mask = self._cache
        stride, kernel_size = self.stride, self.kernel_size
        batch_size, channels, height, width = x_shape
        _, _, height_out, width_out = grad.shape

        dx = np.zeros(x_shape)
        for n in range(batch_size):
            for c in range(channels):
                for i in range(height_out):
                    for j in range(width_out):
                        # h0 - height_stride, w0 - width_stride
                        h0, w0 = i * stride, j * stride

                        # distribute gradient to max positions
                        dx[n, c, h0 : h0 + kernel_size, w0 : w0 + kernel_size] += (
                            mask[n, c, h0 : h0 + kernel_size, w0 : w0 + kernel_size]
                            * grad[n, c, i, j]
                        )
        return dx
