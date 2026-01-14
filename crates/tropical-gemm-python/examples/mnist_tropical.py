#!/usr/bin/env python3
"""
MNIST Classification with Tropical Neural Networks

This example demonstrates using tropical MaxPlus layers in a neural network
for digit classification, compared with standard ReLU networks.

Tropical semiring: C[i,j] = max_k(A[i,k] + B[k,j])
- The "max" provides non-linearity (winner-take-all selection)
- The "+" combines inputs with learned weights (in log-space sense)

Architecture:
- Hybrid MLP: Linear -> ReLU -> TropicalMaxPlus -> LayerNorm -> Linear
- Standard MLP: Linear -> ReLU -> Linear -> ReLU -> Linear

Usage:
    pip install tropical-gemm[torch] torchvision
    python examples/mnist_tropical.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tropical_gemm.pytorch import (
    tropical_maxplus_matmul,
    tropical_maxplus_matmul_gpu,
    GPU_AVAILABLE,
)


class TropicalLinear(nn.Module):
    """
    Tropical MaxPlus linear layer with normalization.

    Computes: y[i] = LayerNorm(max_k(x[k] + weight[k, i]) + bias[i])

    The max operation creates sparse gradients (only argmax contributes),
    so LayerNorm helps stabilize training.

    Args:
        in_features: Size of input
        out_features: Size of output
        use_gpu: If True and GPU available, use CUDA acceleration
    """

    def __init__(self, in_features: int, out_features: int, use_gpu: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_gpu = use_gpu and GPU_AVAILABLE

        # Weight initialization spread out for effective max selection
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gpu:
            out = tropical_maxplus_matmul_gpu(x, self.weight)
        else:
            out = tropical_maxplus_matmul(x, self.weight)
        return self.norm(out + self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, gpu={self.use_gpu}"


class HybridTropicalMLP(nn.Module):
    """
    Hybrid MLP combining standard and tropical layers.

    Architecture:
        Input(784) -> Linear(256) -> ReLU -> TropicalMaxPlus(128) -> Linear(10)

    The initial Linear+ReLU prepares features for the tropical layer,
    which provides a different kind of non-linearity (max selection).
    """

    def __init__(self, use_gpu: bool = True):
        super().__init__()

        # Standard input layer
        self.fc_in = nn.Linear(784, 256)

        # Tropical hidden layer (replaces Linear+ReLU)
        self.tropical = TropicalLinear(256, 128, use_gpu=use_gpu)

        # Standard output layer
        self.fc_out = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_in(x))
        x = self.tropical(x)
        return self.fc_out(x)


class StandardMLP(nn.Module):
    """
    Standard MLP with ReLU for comparison.

    Architecture:
        Input(784) -> Linear(256) -> ReLU -> Linear(128) -> ReLU -> Linear(10)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, test_loader, criterion):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(test_loader), 100.0 * correct / total


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, name="Model"):
    """Train a model and return final test accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining {name}...")
    print("-" * 55)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(
            f"Epoch {epoch:2d}/{epochs}: "
            f"Loss={train_loss:.4f}, "
            f"Train={train_acc:.1f}%, "
            f"Test={test_acc:.1f}%"
        )

    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.1f}s")

    return test_acc


def main():
    print("=" * 60)
    print("MNIST Classification: Tropical vs ReLU Networks")
    print("=" * 60)
    print(f"\nTropical GPU Available: {GPU_AVAILABLE}")

    # Hyperparameters
    batch_size = 128
    epochs = 10
    lr = 0.001

    # Data loading
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Train Hybrid Tropical MLP
    tropical_model = HybridTropicalMLP(use_gpu=GPU_AVAILABLE)
    tropical_acc = train_model(
        tropical_model,
        train_loader,
        test_loader,
        epochs=epochs,
        lr=lr,
        name="Hybrid Tropical MLP",
    )

    # Train Standard MLP for comparison
    standard_model = StandardMLP()
    standard_acc = train_model(
        standard_model,
        train_loader,
        test_loader,
        epochs=epochs,
        lr=lr,
        name="Standard MLP (ReLU)",
    )

    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Hybrid Tropical MLP:  {tropical_acc:.1f}% test accuracy")
    print(f"  Standard MLP (ReLU):  {standard_acc:.1f}% test accuracy")
    print()
    print("Architecture comparison:")
    print("  Tropical: Linear -> ReLU -> TropicalMaxPlus -> LayerNorm -> Linear")
    print("  Standard: Linear -> ReLU -> Linear -> ReLU -> Linear")
    print()
    print("The tropical layer uses max_k(x[k] + w[k]) which provides")
    print("winner-take-all selection with sparse gradient flow.")
    print("=" * 60)


if __name__ == "__main__":
    main()
