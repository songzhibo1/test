"""
EvalGuard — Multi-Dataset/Model Configurations (Section VI-A)

Table IV configurations:
┌───────────┬────────────┬────────┬──────────┬───────────────┐
│ Dataset   │ Model      │ Classes│ Baseline │ Random Guess  │
├───────────┼────────────┼────────┼──────────┼───────────────┤
│ CIFAR-10  │ VGG-11     │ 10     │ 92.4%    │ ~10%          │
│ CIFAR-10  │ ResNet-18  │ 10     │ 93.1%    │ ~10%          │
│ CIFAR-100 │ ResNet-18  │ 100    │ 71.9%    │ ~1%           │
│ ImageNet  │ ResNet-50  │ 1000   │ 76.1%    │ ~0.1%         │
│ AG News   │ RoBERTa    │ 4      │ 94.5%    │ ~25%          │
└───────────┴────────────┴────────┴──────────┴───────────────┘

Note on sub-dominant rank space:
- CIFAR-10 (10 classes): k=4 → C(9,4) rank subsets, moderate diversity
- CIFAR-100 (100 classes): rich sub-dominant space, watermark more robust
- ImageNet (1000 classes): richest, but also most noise in rank ordering
- AG News (4 classes): k=4 means ALL classes are permuted, tightest fit
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ============================================================
# CIFAR-10
# ============================================================

def cifar10_config(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test)
    return trainset, testset, 10


def cifar10_resnet18():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model, "avgpool"


def cifar10_vgg11():
    model = torchvision.models.vgg11(num_classes=10)
    return model, "classifier.3"


# ============================================================
# CIFAR-100
# ============================================================

def cifar100_config(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test)
    return trainset, testset, 100


def cifar100_resnet18():
    model = torchvision.models.resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model, "avgpool"


# ============================================================
# ImageNet (via torchvision pretrained)
# ============================================================

def imagenet_resnet50():
    """
    ResNet-50 on ImageNet-1K.
    Use torchvision pretrained weights for baseline (76.1% top-1).
    """
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    )
    return model, "avgpool"


def imagenet_transforms():
    """Standard ImageNet preprocessing."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return transform_val


# ============================================================
# AG News (RoBERTa) — NLP configuration
# ============================================================

def agnews_config():
    """
    AG News: 4-class news classification.

    Requires: pip install transformers datasets

    Special case for watermark: k=4 means ALL non-top-1 classes
    are permuted. With only 4 classes, k=3 (permute 3 sub-dominant)
    is the maximum useful depth.

    Returns instructions since AG News requires HuggingFace.
    """
    try:
        from transformers import RobertaForSequenceClassification, RobertaTokenizer
        from datasets import load_dataset

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=4
        )

        dataset = load_dataset("ag_news")

        def tokenize(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
            )

        train_data = dataset["train"].map(tokenize, batched=True)
        test_data = dataset["test"].map(tokenize, batched=True)

        return model, tokenizer, train_data, test_data, 4

    except ImportError:
        print("AG News requires: pip install transformers datasets")
        return None


# ============================================================
# Smaller Student Architectures (for distillation experiments)
# ============================================================

def mobilenetv2_cifar10():
    """
    MobileNet-v2 as smaller student (Table VI bottom row).
    Tests watermark retention with capacity-limited surrogate.
    """
    model = torchvision.models.mobilenet_v2(num_classes=10)
    # Adapt for CIFAR-10
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    return model, "features.18"  # Last conv block


# ============================================================
# Configuration Registry
# ============================================================

CONFIGS = {
    "cifar10_resnet18": {
        "dataset_fn": cifar10_config,
        "model_fn": cifar10_resnet18,
        "num_classes": 10,
        "expected_baseline": 93.1,
        "random_guess": 10.0,
        "latent_layer": "avgpool",
        "k": 4,  # Permutation depth
    },
    "cifar10_vgg11": {
        "dataset_fn": cifar10_config,
        "model_fn": cifar10_vgg11,
        "num_classes": 10,
        "expected_baseline": 92.4,
        "random_guess": 10.0,
        "latent_layer": "classifier.3",
        "k": 4,
    },
    "cifar100_resnet18": {
        "dataset_fn": cifar100_config,
        "model_fn": cifar100_resnet18,
        "num_classes": 100,
        "expected_baseline": 71.9,
        "random_guess": 1.0,
        "latent_layer": "avgpool",
        "k": 4,
    },
    "imagenet_resnet50": {
        "dataset_fn": None,  # Requires local ImageNet
        "model_fn": imagenet_resnet50,
        "num_classes": 1000,
        "expected_baseline": 76.1,
        "random_guess": 0.1,
        "latent_layer": "avgpool",
        "k": 4,
    },
}