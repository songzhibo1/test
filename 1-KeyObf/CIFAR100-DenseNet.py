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
# 解决部分服务器无法通过 HTTPS 下载数据集的问题
ssl._create_default_https_context = ssl._create_unverified_context

# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义数据存放的层级目录
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
DATASET_DIR = os.path.join(DATA_DIR, "datasets")

# 自动创建目录（确保文件夹存在）
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# [依据 Theorem 4.2]
# Alpha 保持在 (0.5, 0.9) 之间，避免数值溢出或下溢
ALPHA_RANGE = (0.5, 0.9)

print(f"运行设备: {DEVICE}")
print(f"数据存储路径: {DATASET_DIR}")
print(f"混淆策略: DenseNet121 + KeyDyna (Alpha={ALPHA_RANGE})")


# =========================================================================
# 1. 密钥管理模块 (KeyManager)
# =========================================================================
class KeyManager:
    def __init__(self):
        self.mk = b'MasterKey_DenseNet_Secure_2026'
        self.salt = b'Random_Salt_Cifar_100'

    def derive_user_key(self, uid):
        """为用户生成密钥"""
        msg = f"{uid}_userkey".encode()
        return hmac.new(self.salt, msg, hashlib.sha256).digest()

    def get_layer_factor(self, user_key_bytes, layer_idx):
        """
        [核心防御逻辑]
        - 授权用户 (Valid Key): 返回 r ∈ [10, 50]。
          这会与模型中的 embed_factor (1/(alpha*r)) 结合，
          计算结果为 1/(alpha*r) * r = 1/alpha，从而完美还原特征。

        - 攻击者 (No Key/Wrong Key): 返回 1.0 (或其他随机值)。
          计算结果为 1/(alpha*r) * 1。特征值被缩小了 r 倍 (10~50倍)。
          经过 DenseNet 几十层的累积，特征值会迅速趋近于 0，导致模型失效。
        """
        if user_key_bytes is None:
            return 1.0  # 攻击者没有密钥，盲猜因子为 1.0

        mix_msg = user_key_bytes + f"_densenet_layer_{layer_idx}".encode()
        h = hmac.new(self.mk, mix_msg, hashlib.sha256).digest()
        seed = int.from_bytes(h[:4], 'big')
        rng = np.random.RandomState(seed)

        # 返回强混淆因子 r
        return rng.uniform(10.0, 50.0)


# =========================================================================
# 2. KeyDyna DenseNet 封装类
# =========================================================================
class KeyDynaDenseNet(nn.Module):
    def __init__(self, original_model, alpha_range, correct_uid, key_manager):
        super().__init__()
        # 深拷贝原模型，避免破坏原有的预训练权重
        self.backbone = copy.deepcopy(original_model)
        self.key_manager = key_manager

        print(f"\n   [KeyDyna] 正在初始化 DenseNet121 保护层...")

        # 获取正确用户的根密钥
        correct_uk = key_manager.derive_user_key(correct_uid)

        # ----------------------------------------------------
        # 策略：筛选 DenseNet 所有的 Conv2d 层
        # ----------------------------------------------------
        all_convs = []
        # 递归查找 features 中的所有 Conv2d
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                all_convs.append(module)

        # 选取中间的层进行混淆 (避开首尾，保证梯度稳定性，同时造成中间特征崩塌)
        # 这里我们每隔 2 层混淆一次，覆盖模型中间 60% 的区域
        start_idx = int(len(all_convs) * 0.2)
        end_idx = int(len(all_convs) * 0.8)
        target_layers = all_convs[start_idx: end_idx: 2]

        print(f"   [Config] 共检测到 {len(all_convs)} 个卷积层，选中其中 {len(target_layers)} 个进行加密混淆。")

        for idx, conv_module in enumerate(target_layers):
            # 使用唯一 ID 区分层
            layer_id = idx + 100

            # 1. 生成 Alpha (基于 Theorem 4.2)
            alpha_i = np.random.uniform(alpha_range[0], alpha_range[1])

            # 2. 生成密钥因子 r
            r_key_layer = key_manager.get_layer_factor(correct_uk, layer_id)

            # 3. 实施参数混淆 W' = alpha * W (Theorem 4.1)
            with torch.no_grad():
                conv_module.weight.data *= alpha_i
                if conv_module.bias is not None:
                    conv_module.bias.data *= alpha_i

            # 4. 计算并存储嵌入因子 E = 1 / (alpha * r)
            r_emb_val = 1.0 / (alpha_i * r_key_layer)

            # 注册为 Buffer (不会被当作参数训练，但会随模型保存)
            self.register_buffer(f'emb_{layer_id}', torch.tensor(r_emb_val))

            # 5. 注册 Forward Hook 用于推理时解密
            conv_module.register_forward_hook(self._make_hook(layer_id))

    def _make_hook(self, layer_id):
        def hook_fn(module, input, output):
            # 获取运行时传递的用户密钥
            user_key = getattr(self, 'current_user_key', None)

            # 1. 尝试获取 r (攻击者只能得到 1.0，合法用户得到正确的 r)
            try:
                r_input = self.key_manager.get_layer_factor(user_key, layer_id)
            except:
                r_input = 1.0

            # 2. 获取模型内嵌因子 E
            emb_factor = getattr(self, f'emb_{layer_id}')

            # 3. 恢复计算: Output_new = Output_old * E * r_input
            return output * emb_factor * r_input

        return hook_fn

    def forward(self, x, user_key=None):
        # 将当前密钥暂存到 self 中，供 Hook 函数读取
        self.current_user_key = user_key
        return self.backbone(x)


# =========================================================================
# 3. 实验运行器 (Experiment Runner)
# =========================================================================
class ExperimentRunner:
    def __init__(self):
        self.device = DEVICE
        self.km = KeyManager()
        self.valid_uid = "Admin_User_TJ25504"

    def prepare_data(self):
        print(f"\n>>> [Step 1] 准备 CIFAR-100 数据集...")
        # CIFAR-100 标准均值方差
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

        # 使用配置好的 DATASET_DIR
        print(f"   数据集路径: {DATASET_DIR}")
        try:
            test_set = datasets.CIFAR100(root=DATASET_DIR, train=False, download=True, transform=transform)
            self.test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            print("   数据集准备就绪。")
        except Exception as e:
            print(f"   数据集下载/加载失败: {e}")
            sys.exit(1)

    def load_base_model(self):
        print(f"\n>>> [Step 2] 加载预训练 DenseNet121 (CIFAR-100)...")
        try:
            # 自动从 GitHub 下载 Chenyaofo 的 DenseNet121
            self.base_model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                                             "cifar100_densenet121",
                                             pretrained=True,
                                             verbose=False)
            self.base_model.to(self.device)
            self.base_model.eval()
            print("   原始模型加载成功。")
        except Exception as e:
            print(f"   模型加载失败: {e}")
            print("   请检查网络连接或 PyTorch Hub 缓存。")
            sys.exit(1)

    def evaluate(self, model, user_key=None, desc=""):
        correct = 0
        total = 0
        model.eval()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # 判断是原始模型还是保护模型
                if isinstance(model, KeyDynaDenseNet):
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

        # -------------------------------------------------
        # 1. 基准测试 (Baseline)
        # -------------------------------------------------
        print("\n" + "=" * 50)
        print("1. 基准测试 (Baseline)")
        print("=" * 50)
        base_acc = self.evaluate(self.base_model, desc="Original DenseNet")

        # -------------------------------------------------
        # 2. 构建保护模型 (Encryption)
        # -------------------------------------------------
        print("\n" + "=" * 50)
        print("2. 实施 KeyDyna 保护")
        print("=" * 50)
        self.protected_model = KeyDynaDenseNet(
            self.base_model,
            alpha_range=ALPHA_RANGE,
            correct_uid=self.valid_uid,
            key_manager=self.km
        ).to(self.device)

        # -------------------------------------------------
        # 3. 授权访问测试 (Authorized Access)
        # -------------------------------------------------
        valid_key = self.km.derive_user_key(self.valid_uid)
        auth_acc = self.evaluate(self.protected_model, user_key=valid_key, desc="授权访问 (Correct Key)")

        diff = abs(base_acc - auth_acc)
        if diff < 1.0:
            print(f"   ✅ [Pass] 授权精度正常 (偏差 {diff:.2f}%)，模型功能完整。")
        else:
            print(f"   ⚠️ [Info] 精度偏差 {diff:.2f}% (可能是浮点误差或混淆层数过多)。")

        # -------------------------------------------------
        # 4. 攻击测试 (Security / Confusion Check)
        # -------------------------------------------------
        print("\n" + "=" * 50)
        print("3. 攻击测试 (Security Check)")
        print("=" * 50)

        # 场景 A: 无密钥直接推理
        no_key_acc = self.evaluate(self.protected_model, user_key=None, desc="无密钥访问 (No Key)")

        # 场景 B: 使用错误密钥
        wrong_key = self.km.derive_user_key("Hacker_Bob")
        wrong_acc = self.evaluate(self.protected_model, user_key=wrong_key, desc="错误密钥 (Wrong Key)")

        print("-" * 30)
        # CIFAR-100 随机猜测准确率为 1%
        if no_key_acc < 2.0 and wrong_acc < 2.0:
            print(f"   ✅ [Success] 防御极其有效！准确率已降至随机水平 (~1%)。")
            print(f"   原因: 特征图数值被严重压缩，导致分类器失效。")
        else:
            print(f"   ⚠️ 混淆效果不足，当前非法访问精度: {no_key_acc}%")


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()