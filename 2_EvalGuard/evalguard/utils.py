"""
EvalGuard — Utility Functions

Common helpers used across experiments:
- Model training with logging
- Accuracy evaluation
- Result formatting and saving
- Parameter counting
"""
from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int = 50,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    """Standard SGD training with cosine annealing."""
    model.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
        scheduler.step()

        if verbose and (epoch + 1) % 10 == 0:
            acc = correct / total * 100
            print(f"  Epoch {epoch+1:3d}/{epochs}, "
                  f"Loss: {running_loss/len(trainloader):.4f}, "
                  f"Train Acc: {acc:.2f}%")

    return model


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    testloader: DataLoader,
    device: str = "cpu",
) -> float:
    """Compute top-1 accuracy."""
    model.to(device).eval()
    correct, total = 0, 0
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs).argmax(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return correct / total


@torch.no_grad()
def evaluate_top5_accuracy(
    model: nn.Module,
    testloader: DataLoader,
    device: str = "cpu",
) -> float:
    """Compute top-5 accuracy (for ImageNet)."""
    model.to(device).eval()
    correct, total = 0, 0
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, top5 = model(inputs).topk(5, dim=1)
        correct += top5.eq(targets.unsqueeze(1)).any(1).sum().item()
        total += targets.size(0)
    return correct / total


def count_parameters(model: nn.Module) -> Dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def save_results(results: Dict, path: str = "results.json"):
    """Save experiment results to JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def load_results(path: str = "results.json") -> Dict:
    """Load experiment results from JSON."""
    with open(path) as f:
        return json.load(f)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.label:
            print(f"  [{self.label}] {self.elapsed:.3f}s")


def format_table(headers: list, rows: list, title: str = "") -> str:
    """Format data as ASCII table for console output."""
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

    lines = []
    if title:
        lines.append(title)
    lines.append(sep)
    lines.append(header_line)
    lines.append(sep)
    for row in rows:
        line = "|" + "|".join(f" {str(row[i]):<{col_widths[i]}} " for i in range(len(headers))) + "|"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)
