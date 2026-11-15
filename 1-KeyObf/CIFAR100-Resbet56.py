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

# =========================================================================
# 0. 全局配置 (Configuration)
# =========================================================================
# 解决服务器下载数据集时的 SSL 证书报错
ssl._create_default_https_context = ssl._create_unverified_context

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
DATASET_DIR = os.path.join(DATA_DIR, "datasets")

# 自动创建目录
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# --- 实验参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
ALPHA_RANGE = (0.5, 0.9)  # 混淆系数范围 (Theorem 4.2)

print(f"运行设备: {DEVICE}")
print(f"数据路径: {DATASET_DIR}")
print(f"混淆策略: KeyDyna (ResNet56 + Alpha{ALPHA_RANGE})")


# =========================================================================
# 1. 密钥管理模块 (KeyManager)
# =========================================================================
class KeyManager:
    def __init__(self):
        self.mk = b'MasterKey_ResNet_Secure_2026'
        self.salt = b'Random_Salt_Cifar_100_V2'

    def derive_user_key(self, uid):
        """为用户生成密钥"""
        msg = f"{uid}_userkey".encode()
        return hmac.new(self.salt, msg, hashlib.sha256).digest()

    def get_layer_factor(self, user_key_bytes, layer_idx):
        """
        [安全核心逻辑]
        - 授权用户: 返回 r ∈ [10, 50] (用于抵消嵌入因子的分母)
        - 非法用户: 返回 1.0 (导致特征值被除以 r，即缩小 10~50 倍)
        """
        if user_key_bytes is None:
            return 1.0  # 攻击者盲猜因子

        mix_msg = user_key_bytes + f"_resnet_layer_{layer_idx}".encode()
        h = hmac.new(self.mk, mix_msg, hashlib.sha256).digest()
        seed = int.from_bytes(h[:4], 'big')
        rng = np.random.RandomState(seed)

        return rng.uniform(10.0, 50.0)


# =========================================================================
# 2. KeyDyna 通用模型封装器
# =========================================================================
class KeyDynaWrapper(nn.Module):
    def __init__(self, original_model, alpha_range, correct_uid, key_manager):
        super().__init__()
        self.backbone = copy.deepcopy(original_model)
        self.key_manager = key_manager

        print(f"\n   [KeyDyna] 正在扫描模型结构并注入混淆...")

        # 获取正确用户的根密钥
        correct_uk = key_manager.derive_user_key(correct_uid)

        # --- 自动扫描所有 Conv2d 层 ---
        all_convs = []
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                all_convs.append(module)

        # 混淆策略：每隔 1 层混淆一次，覆盖全模型
        # ResNet56 层数较多，我们混淆其中约 20-30 层
        start_idx = 1
        end_idx = len(all_convs) - 1
        target_layers = all_convs[start_idx: end_idx: 2]

        print(f"   [Config] 模型共有 {len(all_convs)} 个卷积层，选中 {len(target_layers)} 个进行加密。")

        for idx, conv_module in enumerate(target_layers):
            layer_id = idx + 1000  # 唯一ID

            # 1. 生成 Alpha
            alpha_i = np.random.uniform(alpha_range[0], alpha_range[1])

            # 2. 生成正确密钥因子 r
            r_key_layer = key_manager.get_layer_factor(correct_uk, layer_id)

            # 3. 参数混淆 W' = alpha * W
            with torch.no_grad():
                conv_module.weight.data *= alpha_i
                if conv_module.bias is not None:
                    conv_module.bias.data *= alpha_i

            # 4. 计算嵌入因子 E = 1 / (alpha * r)
            # 存入 buffer 随模型保存
            r_emb_val = 1.0 / (alpha_i * r_key_layer)
            self.register_buffer(f'emb_{layer_id}', torch.tensor(r_emb_val))

            # 5. 注册 Hook 进行运行时解密
            conv_module.register_forward_hook(self._make_hook(layer_id))

    def _make_hook(self, layer_id):
        def hook_fn(module, input, output):
            user_key = getattr(self, 'current_user_key', None)

            # 尝试获取 r (攻击者 -> 1.0, 合法用户 -> r)
            try:
                r_input = self.key_manager.get_layer_factor(user_key, layer_id)
            except:
                r_input = 1.0

            # 获取嵌入因子 E
            emb_factor = getattr(self, f'emb_{layer_id}')

            # 恢复: Output * E * r_input
            return output * emb_factor * r_input

        return hook_fn

    def forward(self, x, user_key=None):
        self.current_user_key = user_key
        return self.backbone(x)


# =========================================================================
# 3. 实验运行器
# =========================================================================
class ExperimentRunner:
    def __init__(self):
        self.device = DEVICE
        self.km = KeyManager()
        self.valid_uid = "Admin_User_TJ25504"

    def prepare_data(self):
        print(f"\n>>> [Step 1] 准备 CIFAR-100 数据集...")
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

        try:
            # 使用 DATASET_DIR
            test_set = datasets.CIFAR100(root=DATASET_DIR, train=False, download=True, transform=transform)
            self.test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            print("   数据集准备就绪。")
        except Exception as e:
            print(f"   数据集下载失败: {e}")
            sys.exit(1)

    def load_base_model(self):
        print(f"\n>>> [Step 2] 加载预训练 ResNet56 (CIFAR-100)...")
        try:
            # 修正：使用 chenyaofo 仓库中存在的 'cifar100_resnet56'
            self.base_model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                                             "cifar100_resnet56",
                                             pretrained=True,
                                             verbose=False)
            self.base_model.to(self.device)
            self.base_model.eval()
            print("   原始模型加载成功。")
        except Exception as e:
            print(f"   模型加载失败: {e}")
            print("   (提示: 请确保网络畅通，或尝试手动下载权重)")
            sys.exit(1)

    def evaluate(self, model, user_key=None, desc=""):
        correct = 0
        total = 0
        model.eval()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if isinstance(model, KeyDynaWrapper):
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
        self.load_base_model()

        # 1. 基准测试
        print("\n" + "=" * 50)
        print("1. 基准测试 (Baseline)")
        print("=" * 50)
        base_acc = self.evaluate(self.base_model, desc="Original ResNet56")

        # 2. 加密混淆
        print("\n" + "=" * 50)
        print("2. 实施 KeyDyna 保护")
        print("=" * 50)
        self.protected_model = KeyDynaWrapper(
            self.base_model,
            alpha_range=ALPHA_RANGE,
            correct_uid=self.valid_uid,
            key_manager=self.km
        ).to(self.device)

        # 3. 授权测试
        valid_key = self.km.derive_user_key(self.valid_uid)
        auth_acc = self.evaluate(self.protected_model, user_key=valid_key, desc="授权访问 (Correct Key)")

        if abs(base_acc - auth_acc) < 1.0:
            print("   ✅ [Pass] 授权精度正常，功能无损。")
        else:
            print(f"   ⚠️ [Info] 精度存在细微偏差: {abs(base_acc - auth_acc):.2f}%")

        # 4. 攻击测试
        print("\n" + "=" * 50)
        print("3. 攻击测试 (Security Check)")
        print("=" * 50)

        no_key_acc = self.evaluate(self.protected_model, user_key=None, desc="无密钥访问 (No Key)")

        wrong_key = self.km.derive_user_key("Hacker")
        wrong_acc = self.evaluate(self.protected_model, user_key=wrong_key, desc="错误密钥 (Wrong Key)")

        print("-" * 30)
        if no_key_acc < 5.0 and wrong_acc < 5.0:
            print(f"   ✅ [Success] 防御极其有效！精度已降至随机水平 (<5%)。")
        else:
            print(f"   ⚠️ 当前非法访问精度: {no_key_acc}%")


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()