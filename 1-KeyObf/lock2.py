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
# 依据 Theorem 4.2: alpha < 1 避免 ReLU6 截断
# 范围 (0.5, 0.9) 既保证混淆力度，又保留数值精度
ALPHA_RANGE = (0.5, 0.9)

print(f"运行设备: {DEVICE}")
print(f"混淆配置: Alpha范围 {ALPHA_RANGE}, 多层累积 + 独立密钥因子")


# =========================================================================
# 1. 密钥管理模块 (KeyManager) - 增强版
# =========================================================================
# =========================================================================
# 1. 密钥管理模块 (KeyManager) - 最终修正版
# =========================================================================
class KeyManager:
    def __init__(self):
        self.mk = b'MasterKey_Final_Secure_2025'
        self.salt = b'Random_Salt_Secure_999'

    def derive_user_key(self, uid):
        msg = f"{uid}_userkey".encode()
        return hmac.new(self.salt, msg, hashlib.sha256).digest()

    def get_layer_factor(self, user_key_bytes, layer_idx):
        """
        [核心修正] 将因子范围迁移至 (10, 50)，彻底避开 1.0。

        原理：
        如果 user_key_bytes 为 None (无密钥)，返回 1.0。
        正确因子 r 在 (10, 50) 之间。
        无密钥恢复系数 = 1.0 / r  ≈ 0.02 ~ 0.1。
        这意味着残差分支的信号被衰减了 10~50 倍，特征被彻底抑制。
        """
        if user_key_bytes is None:
            return 1.0  # 攻击者盲猜 1.0

        mix_msg = user_key_bytes + f"_layer_{layer_idx}".encode()
        h = hmac.new(self.mk, mix_msg, hashlib.sha256).digest()
        seed = int.from_bytes(h[:4], 'big')
        rng = np.random.RandomState(seed)

        # [PDF 引用] 依据参考文献建议，使用 10~50 的大幅度因子
        return rng.uniform(10.0, 50.0)

# =========================================================================
# 2. KeyDyna 模型封装 (KeyDynaModel) - 最终版
# =========================================================================
class KeyDynaMobileNetV2(nn.Module):
    def __init__(self, original_model, alpha_range, correct_uid, key_manager):
        super().__init__()
        self.backbone = copy.deepcopy(original_model)
        self.key_manager = key_manager

        print(f"   [KeyDyna] 初始化保护机制...")
        print(f"   [Strategy] 多层累积误差 (Accumulative Error) + 每层独立密钥 (Per-Layer Keys)")

        # 混淆 MobileNetV2 中间的 10 个 Block (层层破坏残差结构)
        self.target_indices = range(5, 15)
        self.hooks = []

        # 获取正确用户的密钥 (Base Key)
        correct_uk = key_manager.derive_user_key(correct_uid)

        # 调试信息记录
        self.layer_alphas = {}
        self.layer_r_keys = {}

        for idx in self.target_indices:
            block = self.backbone.features[idx]

            # 定位 Projection Layer (最后一个 Conv2d，纯线性)
            convs = [m for m in block.modules() if isinstance(m, nn.Conv2d)]
            if not convs: continue
            target_conv = convs[-1]

            # [核心 1] 独立采样 Alpha
            alpha_i = np.random.uniform(alpha_range[0], alpha_range[1])
            self.layer_alphas[idx] = alpha_i

            # [核心 2] 独立派生该层的正确 r_key
            # 即使攻击者猜对了某一层的 1.0，也无法猜对所有层
            r_key_layer = key_manager.get_layer_factor(correct_uk, idx)
            self.layer_r_keys[idx] = r_key_layer

            # [核心 3] 参数混淆 (W' = alpha * W)
            with torch.no_grad():
                target_conv.weight.data *= alpha_i
                if target_conv.bias is not None:
                    target_conv.bias.data *= alpha_i

            # [核心 4] 计算并存储嵌入因子
            # r_emb = 1 / (alpha_i * r_key_layer)
            # 只有当用户输入正确的 r_key_layer 时，乘积才为 1/alpha_i，才能还原特征
            r_emb_val = 1.0 / (alpha_i * r_key_layer)
            self.register_buffer(f'emb_factor_{idx}', torch.tensor(r_emb_val))

            # [核心 5] 注册 Hook
            target_conv.register_forward_hook(self._make_hook(idx))

        print(f"   [Config] 已混淆 {len(self.target_indices)} 层。")
        print(f"   [Debug] Layer 5: Alpha={self.layer_alphas[5]:.3f}, r_key={self.layer_r_keys[5]:.3f}")
        print(f"   [Debug] Layer 6: Alpha={self.layer_alphas[6]:.3f}, r_key={self.layer_r_keys[6]:.3f}")

    def _make_hook(self, layer_idx):
        """闭包 Hook，动态获取该层的因子"""

        def hook_fn(module, input, output):
            user_key = getattr(self, 'current_user_key', None)

            # 动态获取该层专属的密钥因子
            try:
                r_input = self.key_manager.get_layer_factor(user_key, layer_idx)
            except:
                r_input = 1.0

            # 获取该层固化的嵌入因子
            emb_factor = getattr(self, f'emb_factor_{layer_idx}')

            # 执行恢复: Output * emb * r_input
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
        try:
            test_set = datasets.CIFAR100(root=DATASET_DIR, train=False, download=True, transform=transform)
            self.test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            print("   数据集加载完毕。")
        except Exception as e:
            print(f"   数据集下载失败: {e}")
            sys.exit(1)

    def load_original_model(self):
        print(f"\n>>> [Step 2] 加载预训练 MobileNetV2...")
        try:
            self.original_model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models",
                "cifar100_mobilenetv2_x1_0",
                pretrained=True,
                verbose=False
            )
            self.original_model.to(self.device)
            self.original_model.eval()
            print("   模型加载成功。")
        except Exception as e:
            print(f"   模型加载失败: {e}")
            sys.exit(1)

    def evaluate(self, model, user_key=None, desc=""):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if isinstance(model, KeyDynaMobileNetV2):
                    outputs = model(images, user_key=user_key)
                else:
                    outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"   [{desc}] Accuracy: {acc:.2f}%")
        return acc

    def run(self):
        self.prepare_data()
        self.load_original_model()

        # 基准测试
        print("\n>>> [Baseline] 测试原始模型精度...")
        baseline_acc = self.evaluate(self.original_model, desc="Original Model")

        # 构建混淆模型
        print("\n>>> [KeyDyna] 生成受保护模型...")
        self.protected_model = KeyDynaMobileNetV2(
            self.original_model,
            alpha_range=ALPHA_RANGE,
            correct_uid=self.valid_uid,
            key_manager=self.km
        ).to(self.device)
        self.protected_model.eval()

        # 实验 1: 功能一致性
        print("\n" + "=" * 50)
        print("实验 1: 功能一致性验证 (Fidelity)")
        print("=" * 50)
        valid_key = self.km.derive_user_key(self.valid_uid)
        auth_acc = self.evaluate(self.protected_model, user_key=valid_key, desc="KeyDyna (授权用户)")

        if abs(baseline_acc - auth_acc) < 0.1:
            print("   ✅ [PASS] 授权推理精度无损。")
        else:
            print("   ❌ [FAIL] 精度损失过大。")

        # 实验 2: 访问控制
        print("\n" + "=" * 50)
        print("实验 2: 访问控制有效性验证 (Access Control)")
        print("=" * 50)

        # A. 错误密钥
        print("   [测试场景 A] 伪造密钥 (Wrong Key)...")
        attacker_key = self.km.derive_user_key("Hacker_Bob")
        wrong_acc = self.evaluate(self.protected_model, user_key=attacker_key, desc="KeyDyna (错误密钥)")

        # B. 无密钥
        print("   [测试场景 B] 无密钥直接调用 (No Key)...")
        no_key_acc = self.evaluate(self.protected_model, user_key=None, desc="KeyDyna (无密钥)")

        print("-" * 30)
        # 预期：无论怎么猜，精度都应该在 1% ~ 5% 左右
        if wrong_acc < 10.0 and no_key_acc < 10.0:
            print(f"   ✅ [PASS] 保护极其有效 (Wrong: {wrong_acc:.2f}%, No Key: {no_key_acc:.2f}%)。")
            print("   结论: 独立层密钥因子成功封堵了盲猜攻击漏洞。")
        else:
            print(f"   ⚠️ [WARNING] 精度仍偏高 (No Key: {no_key_acc:.2f}%)，请检查混淆层数。")

        print("=" * 50)


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()