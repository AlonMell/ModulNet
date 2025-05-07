# ModulNet

A lightweight neural network framework written from scratch in NumPy.
Includes core building blocks for fully-connected and convolutional networks, custom optimizers, loss functions, and data loading utilities.

## Explanation
![Architecture](MLP.svg)

## Features

- Module API with `forward`/`backward` methods
- Layers: `Linear`, `Conv2D`, `MaxPool2D`, `Flatten`, `ReLU`, `Sigmoid`, `DropOut`
- Loss: `CrossEntropy` with softmax
- Optimizers: `SGD`, `Adagrad`, `RMSProp`, `Adam`
- Regularization utilities: `L1`, `L2`, `ElasticNet`
- Simple `DataLoader` for batching
- Example ConvNet training on MNIST in `main.py`
- Jupyter notebooks for experimentation
- Ruff, pytest integration for linting, formatting, and testing

## Installation

```bash
git clone https://github.com/AlonMell/ModulNet.git module_net
cd module_net
pip install -r requirements.txt
```

## Usage

Train the example convolutional network on MNIST:

```bash
python main.py
```

Or explore the ModuleNet implementation in `notebooks/module.ipynb`.

## Development

- Lint: `make lint`
- Format: `make fmt`
- Run tests: `make test` or `pytest`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).