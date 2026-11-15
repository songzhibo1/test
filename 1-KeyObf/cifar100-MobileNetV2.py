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
import sys
import time

# 解决 SSL 证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# =========================================================================
# 0. 全局配置 (Configuration)
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
DATASET_DIR = os.path.join(DATA_DIR, "datasets")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100

# [关键参数]
ALPHA_RANGE = (0.5, 0.9)

print(f"运行设备: {DEVICE}")
print(f"混淆配置: Alpha范围 {ALPHA_RANGE}, 多层累积 + 独立密钥因子")


# =========================================================================
# 1. 密钥管理模块 (KeyManager)
# =========================================================================
class KeyManager:
    def __init__(self):
        self.mk = b'MasterKey_Final_Secure_2025'
        self.salt = b'Random_Salt_Secure_999'

    def derive_user_key(self, uid):
        msg = f"{uid}_userkey".encode()
        return hmac.new(self.salt, msg, hashlib.sha256).digest()

    def get_layer_factor(self, user_key_bytes, layer_idx):
        if user_key_bytes is None:
            return 1.0

        mix_msg = user_key_bytes + f"_layer_{layer_idx}".encode()
        h = hmac.new(self.mk, mix_msg, hashlib.sha256).digest()
        seed = int.from_bytes(h[:4], 'big')
        rng = np.random.RandomState(seed)
        return rng.uniform(10.0, 50.0)


# =========================================================================
# 2. KeyDyna 模型封装 (KeyDynaModel)
# =========================================================================
class KeyDynaMobileNetV2(nn.Module):
    def __init__(self, original_model, alpha_range, correct_uid, key_manager):
        super().__init__()
        self.backbone = copy.deepcopy(original_model)
        self.key_manager = key_manager

        print(f"   [KeyDyna] 初始化保护机制...")
        print(f"   [Strategy] 多层累积误差 + 每层独立密钥")

        self.target_indices = range(5, 15)
        correct_uk = key_manager.derive_user_key(correct_uid)

        for idx in self.target_indices:
            block = self.backbone.features[idx]
            convs = [m for m in block.modules() if isinstance(m, nn.Conv2d)]
            if not convs: continue
            target_conv = convs[-1]

            alpha_i = np.random.uniform(alpha_range[0], alpha_range[1])
            r_key_layer = key_manager.get_layer_factor(correct_uk, idx)

            with torch.no_grad():
                target_conv.weight.data *= alpha_i
                if target_conv.bias is not None:
                    target_conv.bias.data *= alpha_i

            r_emb_val = 1.0 / (alpha_i * r_key_layer)
            self.register_buffer(f'emb_factor_{idx}', torch.tensor(r_emb_val))
            target_conv.register_forward_hook(self._make_hook(idx))

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            user_key = getattr(self, 'current_user_key', None)
            try:
                r_input = self.key_manager.get_layer_factor(user_key, layer_idx)
            except:
                r_input = 1.0
            emb_factor = getattr(self, f'emb_factor_{layer_idx}')
            return output * emb_factor * r_input

        return hook_fn

    def forward(self, x, user_key=None):
        self.current_user_key = user_key
        out = self.backbone(x)
        return out


# =========================================================================
# 3. 实验运行器 (Experiment Runner)
# =========================================================================
class ExperimentRunner:
    def __init__(self):
        self.device = DEVICE
        self.km = KeyManager()
        self.valid_uid = "Authorized_User_Alice"

    def prepare_data(self):
        print(f"\n>>> [Step 1] 准备 CIFAR-100 数据集...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])
        test_set = datasets.CIFAR100(root=DATASET_DIR, train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"   数据集加载完毕。总样本数: {len(test_set)}")

    def load_original_model(self):
        print(f"\n>>> [Step 2] 加载预训练 MobileNetV2...")
        self.original_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_0",
                                             pretrained=True, verbose=False)
        self.original_model.to(self.device).eval()
        print("   模型加载成功。")

    def evaluate(self, model, user_key=None, desc=""):
        correct = 0
        total = 0
        total_inference_ms = 0.0

        # Warm-up (预热)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32).to(self.device)
            _ = model(dummy, user_key=user_key) if isinstance(model, KeyDynaMobileNetV2) else model(dummy)

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # 计时开始
                if self.device.type == 'cuda': torch.cuda.synchronize()
                t_start = time.perf_counter()

                if isinstance(model, KeyDynaMobileNetV2):
                    outputs = model(images, user_key=user_key)
                else:
                    outputs = model(images)

                # 计时结束
                if self.device.type == 'cuda': torch.cuda.synchronize()
                t_end = time.perf_counter()

                # 累加耗时 (转换为毫秒)
                total_inference_ms += (t_end - t_start) * 1000

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        # 计时推算 (单位：ms)
        ms_for_2000 = (total_inference_ms / total) * 2000
        avg_per_img_ms = total_inference_ms / total

        print(f"   [{desc}] Accuracy: {acc:.2f}%")
        print(f"      [Time] 测试全量({total}张)总推理耗时: {total_inference_ms:.2f} ms")
        print(f"      [Time] 推算2000张图片总耗时: {ms_for_2000:.2f} ms")
        print(f"      [Time] 平均单张图片耗时: {avg_per_img_ms:.4f} ms")

        return {"acc": acc, "time_2000_ms": ms_for_2000}

    def run(self):
        self.prepare_data()
        self.load_original_model()

        # 基准测试
        print("\n>>> [Baseline] 测试原始模型性能...")
        orig_res = self.evaluate(self.original_model, desc="Original Model")

        # 构建混淆模型
        print("\n>>> [KeyDyna] 生成受保护模型...")
        self.protected_model = KeyDynaMobileNetV2(self.original_model, ALPHA_RANGE, self.valid_uid, self.km).to(
            self.device)
        self.protected_model.eval()

        # 实验 1: 一致性与计时
        print("\n" + "=" * 50)
        print("实验 1: 功能一致性验证 与 推理开销分析")
        print("=" * 50)
        valid_key = self.km.derive_user_key(self.valid_uid)
        auth_res = self.evaluate(self.protected_model, user_key=valid_key, desc="KeyDyna (授权用户)")

        # 性能开销总结
        overhead = ((auth_res['time_2000_ms'] / orig_res['time_2000_ms']) - 1) * 100
        print(f"\n   [Summary] 性能开销 (基于2000张推算):")
        print(f"      原始总耗时: {orig_res['time_2000_ms']:.2f} ms")
        print(f"      授权总耗时: {auth_res['time_2000_ms']:.2f} ms")
        print(f"      DynaMO 带来的额外延迟: {overhead:.2f}%")

        # 实验 2: 访问控制 (原本打印不改)
        print("\n" + "=" * 50)
        print("实验 2: 访问控制有效性验证 (Access Control)")
        print("=" * 50)
        attacker_key = self.km.derive_user_key("Hacker_Bob")
        wrong_acc = self.evaluate(self.protected_model, user_key=attacker_key, desc="KeyDyna (错误密钥)")['acc']
        no_key_acc = self.evaluate(self.protected_model, user_key=None, desc="KeyDyna (无密钥)")['acc']

        print("-" * 30)
        if wrong_acc < 10.0 and no_key_acc < 10.0:
            print(f"   ✅ [PASS] 保护极其有效 (Wrong: {wrong_acc:.2f}%, No Key: {no_key_acc:.2f}%)。")
        else:
            print(f"   ⚠️ [WARNING] 精度仍偏高。")
        print("=" * 50)


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()