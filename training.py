"""Shared data loading, training, and checkpoint helpers."""

import argparse
import math
import random
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def add_training_arguments(parser, default_epochs=5):
    parser.add_argument("--dataset", choices=("cifar10", "mnist"), default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=positive_int, default=default_epochs)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--lr", type=positive_float, default=1e-3)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="optional checkpoint destination")
    parser.add_argument(
        "--max-batches", type=positive_int,
        help="limit batches per epoch and evaluation for a quick smoke run",
    )


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested="auto"):
    mps_available = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu or auto")
    if requested == "mps" and not mps_available:
        raise ValueError("MPS was requested but is unavailable; use --device cpu or auto")
    return torch.device(requested)


def make_loaders(args, train_transform, test_transform):
    dataset_type = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.MNIST
    train_set = dataset_type(
        root=str(args.data_dir), train=True, download=True, transform=train_transform,
    )
    test_set = dataset_type(
        root=str(args.data_dir), train=False, download=True, transform=test_transform,
    )
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)
    return train_loader, test_loader, list(train_set.classes)


def _run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    batches = loader if max_batches is None else islice(loader, max_batches)
    with torch.set_grad_enabled(training):
        for images, labels in batches:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size
    if total_examples == 0:
        raise ValueError("the data loader did not contain any examples")
    return total_loss / total_examples, total_correct / total_examples


def fit(model, train_loader, test_loader, optimizer, args, device):
    """Optimize with cross entropy and report sample-weighted held-out metrics."""
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, device, optimizer, args.max_batches,
        )
        test_loss, test_accuracy = _run_epoch(
            model, test_loader, criterion, device, max_batches=args.max_batches,
        )
        history.append(dict(
            train_loss=train_loss, train_accuracy=train_accuracy,
            test_loss=test_loss, test_accuracy=test_accuracy,
        ))
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.2%} | "
            f"test loss {test_loss:.4f}, accuracy {test_accuracy:.2%}",
            flush=True,
        )
    return history


def save_checkpoint(model, args, class_names):
    if args.output is None:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "dataset": args.dataset,
        "classes": list(class_names),
        "architecture": type(model).__name__,
    }, args.output)
    print(f"Saved checkpoint to {args.output}")
