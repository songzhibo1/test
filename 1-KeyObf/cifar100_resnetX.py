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
import time  # 引入计时库

# =========================================================================
# 0. 环境与路径配置
# =========================================================================
ssl._create_default_https_context = ssl._create_unverified_context
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)


# =========================================================================
# 1. 核心类 (KeyManager & Protector)
# =========================================================================
class KeyManagementCenter:
    def __init__(self, master_key=b'DynaMO_Master_Secret_2026'):
        self.master_key = master_key

    def derive_params(self, uid, layer_id, alpha_range=(0.05, 0.15)):
        if uid is None: return None
        msg = f"{uid}_{layer_id}".encode()
        h = hmac.new(self.master_key, msg, hashlib.sha256).digest()
        seed = int.from_bytes(h[:4], 'big')
        rng = np.random.RandomState(seed)
        return {"alpha": rng.uniform(alpha_range[0], alpha_range[1]), "fp": rng.uniform(0, 1)}


class DynaMO_Protector(nn.Module):
    def __init__(self, model, admin_uid, kmc):
        super().__init__()
        self.backbone = copy.deepcopy(model)
        self.kmc = kmc
        self.admin_uid = admin_uid
        self._apply_obfuscation()

    def _apply_obfuscation(self):
        for name, m in self.backbone.named_modules():
            if m.__class__.__name__ == 'BasicBlock':
                layer_id = f"block_{name.replace('.', '_')}"
                p = self.kmc.derive_params(self.admin_uid, layer_id)
                a_t, fp_t = p['alpha'], p['fp']

                with torch.no_grad():
                    m.conv1.weight.data *= a_t
                    m.bn1.weight.data *= a_t
                    m.bn1.bias.data *= a_t
                    m.bn1.running_mean.data *= a_t
                    m.bn1.running_var.data *= (a_t ** 2)

                self.register_buffer(f"{layer_id}_fp", torch.tensor(fp_t))
                self.register_buffer(f"{layer_id}_a", torch.tensor(a_t))
                self._inject_gdsogu(m, layer_id)

    def _inject_gdsogu(self, block, layer_id):
        def patched_forward(x):
            identity = x
            out = block.conv1(x)
            out = block.bn1(out)
            out = block.relu(out)

            current_uid = getattr(self, 'current_uid', None)
            u_p = self.kmc.derive_params(current_uid, layer_id)
            target_fp = getattr(self, f"{layer_id}_fp")
            target_a = getattr(self, f"{layer_id}_a")

            if u_p is not None and abs(u_p['fp'] - target_fp.item()) < 1e-7:
                correction = 1.0 / target_a
            else:
                correction = 0.0001

            out = out * correction
            out = block.conv2(out)
            out = block.bn2(out)
            if block.downsample is not None: identity = block.downsample(x)
            out += identity
            return block.relu(out)

        block.forward = patched_forward

    def forward(self, x, uid=None):
        self.current_uid = uid
        return self.backbone(x)


# =========================================================================
# 2. 自动化实验运行逻辑 (带双维度计时)
# =========================================================================
def run_batch_experiments(model_list):
    kmc = KeyManagementCenter()
    admin_uid = "ADMIN_SECURE_TJ2026"
    results = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_loader = DataLoader(datasets.CIFAR100(root=DATASET_DIR, train=False, download=True, transform=transform),
                             batch_size=128, shuffle=False)

    print(f"\n>>> 开始批量 DynaMO 实验。共计 {len(model_list)} 个模型。")

    for model_name in model_list:
        print(f"\n[测试模型: {model_name}]")

        base_model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True).to(DEVICE)
        base_model.eval()

        # 定义带计时功能的测试函数
        def evaluate(m, loader, uid=None):
            correct, total = 0, 0
            pure_inf_ms = 0.0

            # 预热 (Warm-up) 以消除 GPU 初始延迟
            with torch.no_grad():
                dummy = torch.randn(1, 3, 32, 32).to(DEVICE)
                _ = m(dummy, uid=uid) if isinstance(m, DynaMO_Protector) else m(dummy)

            # A. 端到端计时开始 (包含磁盘IO、数据增强、数据拷贝)
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            start_etoe = time.perf_counter()

            with torch.no_grad():
                for imgs, lbls in loader:
                    imgs_gpu = imgs.to(DEVICE)

                    # B. 纯推理计时：开始 (仅计算模型前向传播)
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    t_start = time.perf_counter()

                    outputs = m(imgs_gpu, uid=uid) if isinstance(m, DynaMO_Protector) else m(imgs_gpu)

                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    t_end = time.perf_counter()
                    # B. 纯推理计时：结束

                    pure_inf_ms += (t_end - t_start) * 1000
                    correct += (outputs.argmax(1) == lbls.to(DEVICE)).sum().item()
                    total += lbls.size(0)

            # A. 端到端计时结束
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            end_etoe = time.perf_counter()

            total_etoe_ms = (end_etoe - start_etoe) * 1000
            acc = 100 * correct / total
            return acc, total_etoe_ms, pure_inf_ms

        # 1. 基准测试 (Baseline)
        base_acc, base_etoe, base_pure = evaluate(base_model, test_loader)

        # 2. 注入混淆并测试
        protected_model = DynaMO_Protector(base_model, admin_uid, kmc).to(DEVICE)
        protected_model.eval()

        auth_acc, auth_etoe, auth_pure = evaluate(protected_model, test_loader, uid=admin_uid)
        wrong_acc, _, _ = evaluate(protected_model, test_loader, uid="Hacker_ID")
        none_acc, _, _ = evaluate(protected_model, test_loader, uid=None)

        results.append({
            "name": model_name,
            "base_acc": f"{base_acc:.2f}%",
            "auth_acc": f"{auth_acc:.2f}%",
            "wrong_acc": f"{wrong_acc:.2f}%",
            "none_acc": f"{none_acc:.2f}%",
            "base_pure": f"{base_pure:.2f}ms",
            "auth_pure": f"{auth_pure:.2f}ms",
            "overhead": f"{((auth_pure / base_pure) - 1) * 100:.2f}%"
        })

    # 3. 打印最终对比表
    print("\n" + "=" * 115)
    print(
        f"{'Model Name':<20} | {'Baseline':<10} | {'Auth(OK)':<10} | {'Wrong':<8} | {'No Key':<8} | {'Base Pure':<10} | {'Auth Pure':<10} | {'Overhead'}")
    print("-" * 115)
    for r in results:
        print(
            f"{r['name']:<20} | {r['base_acc']:<10} | {r['auth_acc']:<10} | {r['wrong_acc']:<8} | {r['none_acc']:<8} | {r['base_pure']:<10} | {r['auth_pure']:<10} | {r['overhead']}")
    print("=" * 115)
    print("[注] Overhead 基于 10,000 张图片的纯推理总耗时计算。")


# =========================================================================
# 执行主入口
# =========================================================================
if __name__ == "__main__":
    MODELS_TO_TEST = [
        "cifar100_resnet20",
        "cifar100_resnet32",
        "cifar100_resnet44",
        "cifar100_resnet56"
    ]
    run_batch_experiments(MODELS_TO_TEST)