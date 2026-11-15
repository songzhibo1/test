import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import hmac
import hashlib
import numpy as np
import copy
import ssl
import time
import random

ssl._create_default_https_context = ssl._create_unverified_context

# =========================================================================
# 0. 基础配置
# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)

BATCH_SIZE = 128
MIX_RANGES = [(0.05, 0.15), (0.40, 0.60), (0.80, 0.95)]


# =========================================================================
# 1. 核心混淆模块
# =========================================================================
class KeyManager:
    def __init__(self, master_key=b'DynaMO_VGG_Secret_2026'):
        self.master_key = master_key

    def get_params(self, uid, layer_idx, alpha_range):
        if uid is None: return None
        msg = f"{uid}_layer_{layer_idx}".encode()
        h = hmac.new(self.master_key, msg, hashlib.sha256).digest()
        seed = int.from_bytes(h[:4], 'big')
        rng = np.random.RandomState(seed)
        return {"alpha": rng.uniform(alpha_range[0], alpha_range[1]), "fp": rng.uniform(0, 1)}


class GDSOGU_Layer(nn.Module):
    def __init__(self, original_relu, layer_idx, kmc, target_a, target_fp, alpha_range):
        super().__init__()
        self.relu = original_relu
        self.layer_idx = layer_idx
        self.kmc = kmc
        self.alpha_range = alpha_range
        self.register_buffer('target_a', torch.tensor(target_a))
        self.register_buffer('target_fp', torch.tensor(target_fp))
        self.current_uid = None

    def forward(self, x):
        x = self.relu(x)
        u_p = self.kmc.get_params(self.current_uid, self.layer_idx, self.alpha_range)
        # 严格匹配验证
        if u_p is not None and abs(u_p['fp'] - self.target_fp.item()) < 1e-7:
            correction = 1.0 / self.target_a
        else:
            correction = 0.0001
        return x * correction


class VGG_DynaMO_Protector(nn.Module):
    def __init__(self, model, admin_uid, kmc, alpha_ranges, num_to_obfuscate=8):
        super().__init__()
        self.backbone = copy.deepcopy(model)
        self.kmc = kmc
        self.admin_uid = admin_uid
        self.alpha_ranges = alpha_ranges
        self.num_to_obfuscate = num_to_obfuscate
        self._obfuscate_vgg()

    def _obfuscate_vgg(self):
        layers = self.backbone.features
        conv_idx = 0
        for i in range(len(layers)):
            if isinstance(layers[i], nn.Conv2d):
                if conv_idx < self.num_to_obfuscate:
                    # 修复：确保混淆时的区间选择是可复现的
                    random.seed(conv_idx + 2026)
                    chosen_range = random.choice(self.alpha_ranges)
                    p = self.kmc.get_params(self.admin_uid, conv_idx, chosen_range)
                    a_t, fp_t = p['alpha'], p['fp']

                    with torch.no_grad():
                        layers[i].weight.data *= a_t
                        if layers[i].bias is not None: layers[i].bias.data *= a_t
                        # 核心修复：如果存在 BN 层，必须同步缩放统计量
                        if i + 1 < len(layers) and isinstance(layers[i + 1], nn.BatchNorm2d):
                            bn = layers[i + 1]
                            bn.weight.data *= a_t
                            bn.bias.data *= a_t
                            bn.running_mean.data *= a_t
                            bn.running_var.data *= (a_t ** 2)

                    for j in range(i + 1, len(layers)):
                        if isinstance(layers[j], nn.ReLU):
                            layers[j] = GDSOGU_Layer(layers[j], conv_idx, self.kmc, a_t, fp_t, chosen_range)
                            break
                conv_idx += 1

    def forward(self, x, uid=None):
        for m in self.modules():
            if isinstance(m, GDSOGU_Layer): m.current_uid = uid
        return self.backbone(x)


# =========================================================================
# 2. 评估函数
# =========================================================================
def evaluate(model, loader, uid, desc):
    model.eval()
    correct, total = 0, 0
    pure_inf_ms = 0.0

    with torch.no_grad():
        # 预热
        dummy = torch.randn(1, 3, 32, 32).to(DEVICE)
        _ = model(dummy) if uid == "BASE" else model(dummy, uid=uid)

        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        start_etoe = time.perf_counter()

        for imgs, lbls in loader:
            imgs_gpu = imgs.to(DEVICE)
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model(imgs_gpu) if uid == "BASE" else model(imgs_gpu, uid=uid)

            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            pure_inf_ms += (time.perf_counter() - t0) * 1000
            correct += (outputs.argmax(1) == lbls.to(DEVICE)).sum().item()
            total += lbls.size(0)

        end_etoe = time.perf_counter()

    acc = 100 * correct / total
    etoe_ms = (end_etoe - start_etoe) * 1000
    print(f"   [{desc}] Acc: {acc:.2f}% | EtoE: {etoe_ms:.2f}ms | Pure: {pure_inf_ms:.2f}ms")
    return acc, pure_inf_ms


# =========================================================================
# 3. 主程序
# =========================================================================
if __name__ == "__main__":
    print(f"运行设备: {DEVICE}")
    kmc = KeyManager()
    ADMIN_UID = "VGG_OWNER_TJ"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_loader = DataLoader(datasets.CIFAR100(root=DATASET_DIR, train=False, download=True, transform=transform),
                             batch_size=BATCH_SIZE)

    # 1. 原始模型评估 (Baseline)
    base_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=True).to(
        DEVICE).eval()
    print("\n" + "=" * 80)
    print(f"{'访问场景':<25} | {'性能与精度统计 (N=10000)'}")
    print("=" * 80)
    base_acc, base_time = evaluate(base_model, test_loader, "BASE", "0. 原始模型 (Baseline)")

    # 2. 实施保护并验证
    protected_model = VGG_DynaMO_Protector(base_model, ADMIN_UID, kmc, MIX_RANGES).to(DEVICE)

    evaluate(protected_model, test_loader, ADMIN_UID, "1. 授权合法用户")
    evaluate(protected_model, test_loader, "WRONG_KEY", "2. 错误密钥用户")
    evaluate(protected_model, test_loader, None, "3. 无密钥匿名用户")
    print("=" * 80)