"""
EvalGuard — Dataset/Model Configurations (Section VI-A, Table IV)

[v3] Added stronger teachers for CIFAR-100:
  - ResNet-56 (~74% acc, pretrained via torch.hub)
  - WideResNet-28-10 (~78% acc, manual build, needs training or --pretrained)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# ============================================================
# Data
# ============================================================

def cifar10_data(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    return trainset, testset, DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2), \
           DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


def cifar100_data(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    return trainset, testset, DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2), \
           DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# ============================================================
# Teacher Models
# ============================================================

# --- CIFAR-10 ---
def cifar10_resnet20(pretrained=True):
    print("[INFO] Loading CIFAR-10 ResNet-20 via torch.hub...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=pretrained)
    return model, "layer3"

def cifar10_vgg11(pretrained=True):
    print("[INFO] Loading CIFAR-10 VGG-11-BN via torch.hub...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=pretrained)
    return model, "features"

# --- CIFAR-100 standard ---
def cifar100_resnet20(pretrained=True):
    print("[INFO] Loading CIFAR-100 ResNet-20 via torch.hub...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=pretrained)
    return model, "layer3"

def cifar100_vgg11(pretrained=True):
    print("[INFO] Loading CIFAR-100 VGG-11-BN via torch.hub...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=pretrained)
    return model, "features"

# --- CIFAR-100 stronger teachers ---
def cifar100_resnet56(pretrained=True):
    """ResNet-56 on CIFAR-100: ~74% accuracy."""
    print("[INFO] Loading CIFAR-100 ResNet-56 via torch.hub...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=pretrained)
    return model, "layer3"

def cifar100_wrn2810(pretrained=True):
    """
    WideResNet-28-10 on CIFAR-100: ~78-80% accuracy.
    No pretrained available via hub — must provide --pretrained path or train first.
    """
    print("[INFO] Building CIFAR-100 WideResNet-28-10...")
    if pretrained:
        print("[WARN] No pretrained WRN-28-10 hub weights. Use --pretrained <path> or set pretrained=False to train.")
    model = _build_wide_resnet(depth=28, widen_factor=10, num_classes=100)
    return model, "block3"


# --- WideResNet builder ---
class _WideBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, dropout=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]
        self.conv1 = nn.Conv2d(3, nStages[0], 3, 1, 1, bias=False)
        self.block1 = self._make_layer(nStages[0], nStages[1], n, 1, dropout)
        self.block2 = self._make_layer(nStages[1], nStages[2], n, 2, dropout)
        self.block3 = self._make_layer(nStages[2], nStages[3], n, 2, dropout)
        self.bn = nn.BatchNorm2d(nStages[3])
        self.fc = nn.Linear(nStages[3], num_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride, dropout):
        layers = [_WideBasicBlock(in_ch, out_ch, stride, dropout)]
        for _ in range(1, n_blocks):
            layers.append(_WideBasicBlock(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def _build_wide_resnet(depth=28, widen_factor=10, num_classes=100, dropout=0.3):
    return WideResNet(depth, widen_factor, num_classes, dropout)


# --- ImageNet ---
def imagenet_resnet50(pretrained=True):
    print("[INFO] Loading ImageNet ResNet-50 via torchvision...")
    if pretrained:
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = torchvision.models.resnet50(num_classes=1000)
    return model, "avgpool"

def agnews_roberta(pretrained=True):
    print("[WARN] Text model skipped. (Requires transformers package)")
    return None, None


# ============================================================
# Student Models
# ============================================================

def create_student(num_classes=10, arch="resnet20"):
    """
    Create a randomly initialized student.
    Options: resnet20, resnet56, vgg11, resnet18, mobilenetv2
    """
    if arch == "resnet20":
        hub_name = "cifar{}_resnet20".format(num_classes) if num_classes in (10, 100) else None
        if hub_name:
            return torch.hub.load("chenyaofo/pytorch-cifar-models", hub_name, pretrained=False)
        model = torchvision.models.resnet18(num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        return model

    elif arch == "resnet56":
        hub_name = "cifar{}_resnet56".format(num_classes) if num_classes in (10, 100) else None
        if hub_name:
            return torch.hub.load("chenyaofo/pytorch-cifar-models", hub_name, pretrained=False)
        raise ValueError("resnet56 student only for CIFAR-10/100")

    elif arch == "vgg11":
        hub_name = "cifar{}_vgg11_bn".format(num_classes) if num_classes in (10, 100) else None
        if hub_name:
            return torch.hub.load("chenyaofo/pytorch-cifar-models", hub_name, pretrained=False)
        return torchvision.models.vgg11_bn(num_classes=num_classes)

    elif arch == "resnet18":
        model = torchvision.models.resnet18(num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        return model

    elif arch == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(num_classes=num_classes)
        model.features[0][0] = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        return model

    else:
        raise ValueError("Unknown student arch: {}".format(arch))


# ============================================================
# Configuration Registry
# ============================================================

CONFIGS = {
    "cifar10_resnet20":  {"data_fn": cifar10_data,  "model_fn": cifar10_resnet20,  "num_classes": 10,  "baseline_acc": 92.6, "random_guess": 10.0, "k": 4},
    "cifar10_vgg11":     {"data_fn": cifar10_data,  "model_fn": cifar10_vgg11,     "num_classes": 10,  "baseline_acc": 92.4, "random_guess": 10.0, "k": 4},
    "cifar100_resnet20": {"data_fn": cifar100_data, "model_fn": cifar100_resnet20, "num_classes": 100, "baseline_acc": 68.8, "random_guess": 1.0,  "k": 4},
    "cifar100_vgg11":    {"data_fn": cifar100_data, "model_fn": cifar100_vgg11,    "num_classes": 100, "baseline_acc": 70.0, "random_guess": 1.0,  "k": 4},
    "cifar100_resnet56": {"data_fn": cifar100_data, "model_fn": cifar100_resnet56, "num_classes": 100, "baseline_acc": 74.0, "random_guess": 1.0,  "k": 4},
    "cifar100_wrn2810":  {"data_fn": cifar100_data, "model_fn": cifar100_wrn2810,  "num_classes": 100, "baseline_acc": 78.0, "random_guess": 1.0,  "k": 4},
    "imagenet_resnet50": {"data_fn": None,          "model_fn": imagenet_resnet50, "num_classes": 1000,"baseline_acc": 76.1, "random_guess": 0.1,  "k": 4},
    "agnews_roberta":    {"data_fn": None,          "model_fn": agnews_roberta,    "num_classes": 4,   "baseline_acc": 94.5, "random_guess": 25.0, "k": 3},
}