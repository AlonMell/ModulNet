import os
import sys

import numpy as np

# allow imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from module_net.convolutional import Conv2D, Flatten, MaxPool2D
from module_net.losses import CrossEntropy
from module_net.module import Linear, ReLU, Sequential, Sigmoid


def test_linear_forward_backward():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    linear = Linear(2, 3)
    W = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    b = np.array([0.5, -0.5, 1.0], dtype=float)
    linear._params = [W, b]
    linear._grads = [np.zeros_like(W), np.zeros_like(b)]
    out = linear.forward(x)
    expected = x @ W.T + b
    assert np.allclose(out, expected)
    grad_out = np.ones_like(out)
    dx = linear.backward(grad_out)
    expected_dW = grad_out.T @ x
    expected_db = np.sum(grad_out, axis=0)
    assert np.allclose(linear._grads[0], expected_dW)
    assert np.allclose(linear._grads[1], expected_db)
    assert np.allclose(dx, grad_out @ W)


def test_relu_forward_backward():
    relu = ReLU()
    x = np.array([[-1.0, 0.0, 2.0]])
    out = relu.forward(x)
    assert np.array_equal(out, np.array([[0.0, 0.0, 2.0]]))
    grad = np.array([[1.0, 1.0, 1.0]])
    dx = relu.backward(grad)
    assert np.array_equal(dx, np.array([[0.0, 0.0, 1.0]]))


def test_sigmoid_forward_backward():
    sigmoid = Sigmoid()
    x = np.array([[0.0, 2.0]])
    out = sigmoid.forward(x)
    expected = 1 / (1 + np.exp(-x))
    assert np.allclose(out, expected)
    grad = np.array([[1.0, 1.0]])
    dx = sigmoid.backward(grad)
    deriv = expected * (1 - expected)
    assert np.allclose(dx, deriv)


def test_flatten_forward_backward():
    flatten = Flatten()
    x = np.arange(8).reshape(2, 2, 2)
    out = flatten.forward(x)
    assert out.shape == (2, 4)
    grad = np.arange(8).reshape(2, 2, 2)
    dx = flatten.backward(grad)
    assert dx.shape == x.shape
    assert np.array_equal(dx, grad)


def test_sequential_chain():
    seq = Sequential(Linear(2, 2), ReLU(), Linear(2, 1))
    # override params for predictability
    l0, l1 = seq.layers[0], seq.layers[2]
    l0._params = [np.eye(2), np.zeros(2)]
    l0._grads = [np.zeros((2, 2)), np.zeros(2)]
    l1._params = [np.ones((1, 2)), np.zeros(1)]
    l1._grads = [np.zeros((1, 2)), np.zeros(1)]
    x = np.array([[1.0, -1.0]])
    out = seq.forward(x)
    # after l0: identity * x -> x, relu -> [1,0], l1 -> sum([1,0])=1
    assert np.allclose(out, np.array([[1.0]]))
    grad = np.array([[2.0]])
    dx = seq.backward(grad)
    # backprop through ones and relu mask
    assert dx.shape == x.shape


def test_crossentropy_forward_backward():
    logits = np.array([[2.0, 1.0, 0.1]])
    labels = np.array([[1, 0, 0]])
    ce = CrossEntropy()
    loss = ce.forward(logits, labels)
    exps = np.exp(logits - np.max(logits))
    probs = exps / exps.sum(axis=1, keepdims=True)
    expected_loss = -np.sum(labels * np.log(probs))
    assert np.allclose(loss, expected_loss)
    grad = ce.backward(labels)
    assert np.allclose(grad, (probs - labels) / 1.0)


def test_conv2d_forward_backward():
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    conv = Conv2D(1, 1, kernel_size=2, stride=1, padding=0)
    conv._params = [np.array([[[[1.0, 1.0], [1.0, 1.0]]]]), np.array([0.0])]
    conv._grads = [np.zeros_like(conv._params[0]), np.zeros_like(conv._params[1])]
    out = conv.forward(x)
    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == np.sum(x)
    grad_out = np.array([[[[1.0]]]])
    dx = conv.backward(grad_out)
    assert np.allclose(conv._grads[0], x)
    assert np.allclose(conv._grads[1], np.array([1.0]))
    assert np.allclose(dx, np.ones_like(x))


def test_maxpool2d_forward_backward():
    x = np.array([[[[1.0, 3.0], [2.0, 0.0]]]])
    pool = MaxPool2D(kernel_size=2, stride=2)
    out = pool.forward(x)
    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == 3.0
    grad_out = np.array([[[[1.0]]]])
    dx = pool.backward(grad_out)
    expected = np.array([[[[0.0, 1.0], [0.0, 0.0]]]])
    assert np.array_equal(dx, expected)
