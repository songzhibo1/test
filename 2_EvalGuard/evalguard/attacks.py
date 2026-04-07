"""
EvalGuard — E2 Attack Pipeline (Section VI-C)

[v5] Logit-space watermark embedding:
  - collect_soft_labels now passes raw logits to watermark module
  - Watermark is applied in logit space BEFORE softmax/temperature
  - This makes the watermark T-invariant

1. Soft-label distillation (E2s, Table VI):
   KL divergence on watermarked teacher outputs (Eq. 16)

2. Hard-label extraction (E2h):
   KnockoffNets-style, top-1 only, bypasses watermark

3. Surrogate fine-tuning (Table VII):
   Post-distillation FT to attempt watermark removal
"""
from __future__ import annotations

import copy
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset

from .watermark import WatermarkModule, verify_ownership
from .crypto import keygen, kdf


# ============================================================
# 1. Soft-Label Knowledge Distillation (E2s)
# ============================================================

def collect_soft_labels(
    teacher: nn.Module,
    query_loader: DataLoader,
    device: str = "cpu",
    watermark_module: Optional[WatermarkModule] = None,
    temperature: float = 3.0,
) -> tuple:
    """
    Collect soft-label outputs from TEE (with watermark applied).

    [v5] Watermark is now applied in logit space:
      1. Get raw logits from teacher
      2. Pass logits to watermark module (adds delta_logit to target class)
      3. Watermark module applies softmax(logits / T) internally
      4. Return probabilities as soft labels

    Returns: (inputs, soft_labels, n_watermarked)
    """
    teacher.to(device).eval()
    all_inputs, all_soft_labels = [], []
    n_watermarked = 0

    with torch.no_grad():
        for inputs, _ in query_loader:
            inputs = inputs.to(device)
            logits = teacher(inputs)

            if watermark_module is not None:
                # [v5] Pass raw logits + temperature to watermark module
                # Watermark adds delta in logit space, then applies softmax(logits/T)
                logits_np = logits.cpu().numpy()
                probs_np, n_wm = watermark_module.embed_batch_logits(
                    teacher, inputs, logits_np, temperature, device)
                n_watermarked += n_wm
                probs = torch.from_numpy(probs_np).float().to(device)
            else:
                # No watermark: standard softmax with temperature
                probs = torch.softmax(logits / temperature, dim=-1)

            all_inputs.append(inputs.cpu())
            all_soft_labels.append(probs.cpu())

    return torch.cat(all_inputs, 0), torch.cat(all_soft_labels, 0), n_watermarked


def soft_label_distillation(
    student: nn.Module,
    inputs: torch.Tensor,
    soft_labels: torch.Tensor,
    temperature: float = 3.0,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 0.002,
    device: str = "cpu",
) -> tuple:
    """
    Train surrogate via KL divergence on collected soft labels.
    Loss = T² · KL(softmax(z_teacher/T) || softmax(z_student/T))  [Eq. 16]

    Returns: (student, loss_history)
    """
    dataset = TensorDataset(inputs, soft_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    student.to(device).train()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_soft in loader:
            batch_x, batch_soft = batch_x.to(device), batch_soft.to(device)
            student_log_probs = torch.log_softmax(student(batch_x) / temperature, dim=-1)
            loss = temperature ** 2 * nn.functional.kl_div(
                student_log_probs, batch_soft, reduction="batchmean"
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(loader)
        loss_history.append({"epoch": epoch + 1, "loss": round(avg_loss, 6)})

        if (epoch + 1) % 10 == 0:
            print(f"    Distillation epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return student, loss_history


# ============================================================
# 2. Hard-Label Extraction (E2h)
# ============================================================

def collect_hard_labels(
    teacher: nn.Module,
    query_loader: DataLoader,
    device: str = "cpu",
    watermark_module: Optional[WatermarkModule] = None,
) -> tuple:
    teacher.to(device).eval()
    all_inputs, all_labels = [], []
    n_watermarked = 0 # 硬标签模式下信号丢失，直接设为0

    with torch.no_grad():
        # 这里依然是按 Batch 遍历的 (由 query_loader 控制)
        for inputs, _ in query_loader:
            inputs = inputs.to(device)
            logits = teacher(inputs)

            # 核心优化：直接取 argmax，不要去调用 watermark_module
            # 这样就跳过了那 50,000 次 HMAC 运算
            preds = logits.argmax(dim=-1)

            all_inputs.append(inputs.cpu())
            all_labels.append(preds.cpu())

    return torch.cat(all_inputs, 0), torch.cat(all_labels, 0), n_watermarked
def hard_label_extraction(
    student: nn.Module,
    inputs: torch.Tensor,
    hard_labels: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str = "cpu",
) -> tuple:
    """
    Train surrogate using only hard labels (cross-entropy).
    Returns: (student, loss_history)
    """
    dataset = TensorDataset(inputs, hard_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    student.to(device).train()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = criterion(student(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append({"epoch": epoch + 1, "loss": round(avg_loss, 6)})

        if (epoch + 1) % 10 == 0:
            print(f"    Hard-label epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return student, loss_history


# ============================================================
# 3. Surrogate Fine-Tuning (Table VII)
# ============================================================

def fine_tune_surrogate(
    surrogate: nn.Module,
    ft_loader: DataLoader,
    epochs: int = 20,
    lr: float = 0.0005,
    device: str = "cpu",
) -> tuple:
    """
    Post-distillation fine-tuning to attempt watermark removal.
    Returns: (surrogate, loss_history)
    """
    surrogate.to(device).train()
    optimizer = optim.SGD(surrogate.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in ft_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(surrogate(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(ft_loader)
        loss_history.append({"epoch": epoch + 1, "loss": round(avg_loss, 6)})

        if (epoch + 1) % 10 == 0:
            print(f"    FT epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return surrogate, loss_history