import numpy as np
import torch
import torchvision
from numpy.typing import NDArray
from torchvision import transforms
from tqdm import tqdm, trange

from convolutional import Conv2D, Flatten, MaxPool2D
from losses import CrossEntropy
from module import Linear, Module, ReLU, Sequential
from optimizer import Adam


class ConvNet(Module):
    def __init__(self):
        super().__init__()
        self.net = Sequential(
            Conv2D(in_channels=1, out_channels=6, kernel_size=5),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Conv2D(in_channels=6, out_channels=16, kernel_size=5),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Flatten(),
            Linear(16 * 4 * 4, 120),
            ReLU(),
            Linear(120, 84),
            ReLU(),
            Linear(84, 10),
        )

    def forward(self, x: NDArray) -> NDArray:
        return self.net.forward(x)

    def backward(self, grad: NDArray) -> NDArray:
        return self.net.backward(grad)

    def parameters(self) -> list[tuple[NDArray, NDArray]]:
        return self.net.parameters()

    def zero_grad(self) -> None:
        self.net.zero_grad()


def main():
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=2
    )

    classes = tuple(str(i) for i in range(10))

    print(trainloader.dataset.data.shape)
    print(testloader.dataset.data.shape)

    for input, label in trainloader:
        print(f"input shape: {input.shape}")
        print(f"label shape: {label.shape}")
        break

    model = ConvNet()
    loss_fn = CrossEntropy()
    optimizer = Adam(model.parameters())
    losses = train(model, loss_fn, optimizer, trainloader)
    test(model, testloader)


def train(model, loss_fn, optimizer, trainloader) -> list:
    num_epochs = 2
    losses = []

    for epoch in trange(num_epochs, desc="Epochs"):
        epoch_loss = 0.0
        num_samples = 0

        for x_batch, y_batch in tqdm(
            trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ):
            # x_batch: (batch, 1, 28, 28)
            # y_batch: (batch,)
            x = np.asarray(x_batch.numpy())
            y_idx = y_batch.numpy().astype(int)
            y = np.eye(10, dtype=np.float32)[y_idx]

            logits = model.forward(x)
            loss = loss_fn.forward(logits, y)

            grad = loss_fn.backward(y)

            model.zero_grad()
            model.backward(grad)
            optimizer.step()

            batch_size = x.shape[0]
            epoch_loss += loss * batch_size
            num_samples += batch_size

        avg_loss = epoch_loss / num_samples
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} — avg loss: {avg_loss:.4f}")

    return losses


def test(model, testloader):
    num_classes = 10
    correct_per_class = np.zeros(num_classes, dtype=int)
    total_per_class = np.zeros(num_classes, dtype=int)
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in testloader:
        x = np.asarray(x_batch.numpy())
        y_idx = y_batch.numpy().astype(int)

        logits = model.forward(x)
        preds = np.argmax(logits, axis=1)

        total_samples += y_idx.shape[0]
        total_correct += np.sum(preds == y_idx)

        for pred, label in zip(preds, y_idx):
            total_per_class[label] += 1
            correct_per_class[label] += pred == label

    overall_acc = total_correct / total_samples
    class_acc = correct_per_class / total_per_class

    print(f"Overall accuracy: {overall_acc * 100:.2f}%\n")

    for cls in range(num_classes):
        print(
            f"Class {cls}: {class_acc[cls] * 100:.2f}%  "
            + f"({correct_per_class[cls]}/{total_per_class[cls]})"
        )


if __name__ == "__main__":
    main()
