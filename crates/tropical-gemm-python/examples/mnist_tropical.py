#!/usr/bin/env python3
"""
MNIST Classification with Tropical Neural Networks

This example demonstrates using tropical affine maps in a neural network
for digit classification, compared with standard ReLU networks.

Tropical Affine Map: y[i] = max(max_k(x[k] + W[k,i]), b[i])
- The outer "max" is tropical addition (⊕) combining matmul result with bias
- The inner "max" over k is the tropical matrix multiplication
- The "+" is tropical multiplication (⊗)
- The bias "b" acts as a learned threshold

This is the tropical analog of the standard affine transformation (Ax + b).

Architecture:
- Hybrid MLP: Linear -> TropicalAffine -> LayerNorm -> Linear -> TropicalAffine -> Linear
- Standard MLP: Linear -> ReLU -> Linear -> ReLU -> Linear

Note: This example uses CPU by default. GPU acceleration is more beneficial
for large matrices (1024x1024+). For small matrices like this example,
kernel launch overhead exceeds the computation benefit.

Usage:
    pip install tropical-gemm[torch] torchvision
    python examples/mnist_tropical.py          # CPU mode (faster for small matrices)
    python examples/mnist_tropical.py --gpu    # GPU mode (for demonstration)
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tropical_gemm.pytorch import (
    tropical_maxplus_matmul,
    tropical_minplus_matmul,
    tropical_maxplus_matmul_gpu,
    tropical_minplus_matmul_gpu,
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


class TropicalAffine(nn.Module):
    """
    Tropical MaxPlus affine layer: y = W ⊗ x ⊕ b (in tropical notation).

    Computes: y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])

    LayerNorm is applied before the tropical matmul to stabilize training,
    as max operations create sparse gradients (only argmax contributes).

    The max operation provides winner-take-all non-linearity, while
    the bias provides a learned threshold for each output.

    Args:
        features: Number of features (input = output dimension)
        use_gpu: If True and GPU available, use CUDA acceleration
    """

    def __init__(self, features: int, use_gpu: bool = True):
        super().__init__()
        self.features = features
        self.use_gpu = use_gpu and GPU_AVAILABLE

        # Learnable weight matrix (initialized to near-identity)
        self.norm = nn.LayerNorm(features)
        init_weight = torch.randn((features, features)) * 0.5
        self.weight = nn.Parameter(init_weight)
        # Learnable bias for tropical affine (combined via max)
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tropical matmul: tmp[i] = max_k(x[k] + W[k,i])
        out = self.norm(x)
        if self.use_gpu:
            out = tropical_maxplus_matmul_gpu(out, self.weight)
        else:
            out = tropical_maxplus_matmul(out, self.weight)
        # Tropical affine: y[i] = max(tmp[i], b[i]) - tropical addition with bias
        out = torch.maximum(out, self.bias)
        return out


class HybridTropicalMLP(nn.Module):
    """
    Hybrid MLP using Tropical Affine maps as activation function.

    Architecture:
        Input(784) -> Linear(256) -> TropicalAffine(256)
                   -> Linear(128) -> TropicalAffine(128)
                   -> Linear(10)

    The TropicalAffine layers compute: y[i] = max_k(x[k] + W[k,i]) + b[i]
    This is the tropical analog of an affine transformation, providing
    max-based non-linearity with learnable weights and biases.
    """

    def __init__(self, use_gpu: bool = True):
        super().__init__()

        # Layer 1: Linear + TropicalAffine
        self.fc1 = nn.Linear(784, 256)
        self.act1 = TropicalAffine(256, use_gpu=use_gpu)

        # Layer 2: Linear + TropicalAffine
        self.fc2 = nn.Linear(256, 128)
        self.act2 = TropicalAffine(128, use_gpu=use_gpu)

        # Output layer (no activation - logits for CrossEntropyLoss)
        self.fc_out = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)

        # Layer 1: Linear -> TropicalAffine
        x = self.fc1(x)
        x = self.act1(x)

        # Layer 2: Linear -> TropicalAffine
        x = self.fc2(x)
        x = self.act2(x)

        # Output
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


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
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


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(test_loader), 100.0 * correct / total


def train_model(model, train_loader, test_loader, device, epochs=10, lr=0.001, name="Model"):
    """Train a model and return final test accuracy."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining {name}...")
    print("-" * 55)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

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
    parser = argparse.ArgumentParser(description="MNIST with Tropical Neural Networks")
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use GPU for tropical layers (slower for small matrices due to overhead)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    args = parser.parse_args()

    use_gpu = args.gpu and GPU_AVAILABLE and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    print("=" * 60)
    print("MNIST Classification: Tropical vs ReLU Networks")
    print("=" * 60)
    print(f"\nTropical GPU Available: {GPU_AVAILABLE}")
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if not use_gpu:
        print("(Use --gpu to enable GPU mode)")

    # Hyperparameters
    batch_size = 128
    epochs = args.epochs
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
    tropical_model = HybridTropicalMLP(use_gpu=use_gpu)

    # Record initial weights (clone before moving to GPU)
    init_w1 = tropical_model.act1.weight.data.clone()
    init_w2 = tropical_model.act2.weight.data.clone()

    tropical_acc = train_model(
        tropical_model,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        lr=lr,
        name="Hybrid Tropical MLP",
    )

    # Check if tropical weights changed (move back to CPU for comparison)
    final_w1 = tropical_model.act1.weight.data.cpu()
    final_w2 = tropical_model.act2.weight.data.cpu()

    print("\n--- Tropical Weight Analysis ---")
    print(f"TropicalAffine layer 1 (256x256):")
    print(f"  Weight change (L2 norm): {(final_w1 - init_w1).norm().item():.4f}")
    print(f"  Initial diag mean: {init_w1.diag().mean().item():.4f}, off-diag mean: {((init_w1.sum() - init_w1.diag().sum()) / (256*256-256)).item():.4f}")
    print(f"  Final diag mean:   {final_w1.diag().mean().item():.4f}, off-diag mean: {((final_w1.sum() - final_w1.diag().sum()) / (256*256-256)).item():.4f}")
    print(f"TropicalAffine layer 2 (128x128):")
    print(f"  Weight change (L2 norm): {(final_w2 - init_w2).norm().item():.4f}")
    print(f"  Initial diag mean: {init_w2.diag().mean().item():.4f}, off-diag mean: {((init_w2.sum() - init_w2.diag().sum()) / (128*128-128)).item():.4f}")
    print(f"  Final diag mean:   {final_w2.diag().mean().item():.4f}, off-diag mean: {((final_w2.sum() - final_w2.diag().sum()) / (128*128-128)).item():.4f}")

    # Train Standard MLP for comparison
    standard_model = StandardMLP()
    standard_acc = train_model(
        standard_model,
        train_loader,
        test_loader,
        device,
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
    print("  Tropical: Linear -> TropicalAffine(256) -> Linear -> TropicalAffine(128) -> Linear")
    print("  Standard: Linear -> ReLU -> Linear -> ReLU -> Linear")
    print()
    print("Tropical Affine: y[i] = max(max_k(x[k] + W[k,i]), b[i])")
    print("Bias is combined via max (tropical addition), acts as threshold.")
    print("=" * 60)


if __name__ == "__main__":
    main()
