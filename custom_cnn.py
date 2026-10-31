"""Train a three-convolution CNN on CIFAR-10 or MNIST.

Example: python custom_cnn.py --dataset mnist --epochs 5
"""

import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from training import (
    add_training_arguments,
    fit,
    make_loaders,
    resolve_device,
    save_checkpoint,
    set_seed,
)


class CustomCNN(nn.Module):
    """Three Conv2d/ReLU/MaxPool stages followed by a classification head.

    Inputs must be N x in_channels x 32 x 32. MNIST is resized to 32 x 32
    by the data pipeline and uses in_channels=1.
    """

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))  # 16 -> 8
        x = self.pool(F.relu(self.conv3(x)))  # 8 -> 4
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Raw logits for CrossEntropyLoss.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.dataset == "mnist":
        in_channels = 1
        mean, std = (0.1307,), (0.3081,)
    else:
        in_channels = 3
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_loader, test_loader, classes = make_loaders(args, transform, transform)
    model = CustomCNN(num_classes=len(classes), in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fit(model, train_loader, test_loader, optimizer, args, device)
    save_checkpoint(model, args, classes)


if __name__ == "__main__":
    main()
