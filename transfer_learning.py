"""Train a new classification head on a frozen, pretrained ResNet-18.

Example: python transfer_learning.py --dataset cifar10 --epochs 5
"""

import argparse

import torch
from torch import nn
from torchvision import models, transforms

from training import (
    add_training_arguments,
    fit,
    make_loaders,
    resolve_device,
    save_checkpoint,
    set_seed,
)


class FrozenResNet18(nn.Module):
    def __init__(self, num_classes=10, weights=models.ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()
        # Modern equivalent of torchvision.models.resnet18(pretrained=True).
        # Pass weights=None only for offline architecture tests.
        self.backbone = models.resnet18(weights=weights)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        in_features = self.backbone.fc.in_features
        # New parameters default to requires_grad=True.
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Freezing gradients alone does not freeze BatchNorm running statistics.
        self.backbone.eval()
        self.backbone.fc.train(mode)
        return self

    def forward(self, x):
        return self.backbone(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    # Official preprocessing: resize/crop to 224 x 224 and ImageNet normalization.
    steps = []
    if args.dataset == "mnist":
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps.append(weights.transforms())
    transform = transforms.Compose(steps)
    train_loader, test_loader, classes = make_loaders(args, transform, transform)
    model = FrozenResNet18(num_classes=len(classes), weights=weights).to(device)
    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=args.lr)
    fit(model, train_loader, test_loader, optimizer, args, device)
    save_checkpoint(model, args, classes)


if __name__ == "__main__":
    main()
