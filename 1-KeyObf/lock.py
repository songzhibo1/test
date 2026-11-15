import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import hashlib
import hmac
import struct
import copy
import numpy as np

# =========================================================================
# 0. 全局配置
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
DATASET_DIR = os.path.join(DATA_DIR, "datasets")
LOG_FILE = os.path.join(BASE_DIR, "log_final_correct.txt")

os.environ['TORCH_HOME'] = WEIGHTS_DIR
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# 20 Batch 足够验证，论文数据请设为 None
EVAL_BATCH_SIZE = 100
EVAL_LIMIT_BATCHES = 20


# =========================================================================
# 1. 密码学工具 (KeyManager)
# =========================================================================
class KeyManager:
    @staticmethod
    def get_random_bytes(n):
        return torch.randint(0, 255, (n,), dtype=torch.uint8).numpy().tobytes()

    @staticmethod
    def map_to_float(hash_bytes):
        val = struct.unpack('>Q', hash_bytes[:8])[0]
        return 10.0 + 40.0 * (val / 0xFFFFFFFFFFFFFFFF)

    @staticmethod
    def generate_master_key():
        return KeyManager.get_random_bytes(32), KeyManager.get_random_bytes(16)

    @staticmethod
    def derive_user_key(mk, salt, uid):
        msg = uid.encode() + b"userkey"
        h = hmac.new(salt, msg, hashlib.sha256).digest()
        uk_int = int.from_bytes(mk, 'big') ^ int.from_bytes(h, 'big')
        uk_bytes = uk_int.to_bytes(32, 'big')
        return uk_bytes[-32:] if len(uk_bytes) > 32 else uk_bytes

    @staticmethod
    def recover_runtime_factor(uk, salt, uid):
        msg = uid.encode() + b"userkey"
        h = hmac.new(salt, msg, hashlib.sha256).digest()
        mk_recovered_int = int.from_bytes(uk, 'big') ^ int.from_bytes(h, 'big')
        try:
            mk_recovered = mk_recovered_int.to_bytes(32, 'big')
        except:
            mk_recovered = KeyManager.get_random_bytes(32)
        factor_msg = salt + b"keyfactor"
        factor_hash = hmac.new(mk_recovered, factor_msg, hashlib.sha256).digest()
        return KeyManager.map_to_float(factor_hash)


# =========================================================================
# 2. 非线性耦合模块 (定理 3.2 实现)
# =========================================================================
class NonLinearCoupler(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 线性恢复
        self.coupler = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # 冗余激活
        self.redundant_relu = nn.ReLU(inplace=False)

    def forward(self, x, runtime_key_factor):
        x = self.coupler(x)
        if isinstance(runtime_key_factor, (float, int)):
            x = x * runtime_key_factor
        else:
            x = x * runtime_key_factor.view(1, 1, 1, 1)
        x = self.redundant_relu(x)
        return x


class DynamicLOCKResNet56(nn.Module):
    def __init__(self, original_model, active_points):
        super().__init__()
        self.active_points = active_points

        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.layer1 = copy.deepcopy(original_model.layer1)
        self.layer2 = copy.deepcopy(original_model.layer2)
        self.layer3 = copy.deepcopy(original_model.layer3)
        self.avgpool = original_model.avgpool
        self.fc = copy.deepcopy(original_model.fc)

        self.channel_map = {0: 16, 1: 16, 2: 32, 3: 64}

        self.locks = nn.ModuleDict()
        for p in active_points:
            if p in self.channel_map:
                self.locks[str(p)] = NonLinearCoupler(self.channel_map[p])

    def forward(self, x, runtime_key_factor=1.0):
        # Point 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if '0' in self.locks: x = self.locks['0'](x, runtime_key_factor)

        # Point 1
        x = self.layer1(x)
        if '1' in self.locks: x = self.locks['1'](x, runtime_key_factor)

        # Point 2
        x = self.layer2(x)
        if '2' in self.locks: x = self.locks['2'](x, runtime_key_factor)

        # Point 3
        x = self.layer3(x)
        if '3' in self.locks: x = self.locks['3'](x, runtime_key_factor)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =========================================================================
# 3. 实验运行逻辑
# =========================================================================
class Logger:
    @staticmethod
    def log(message):
        print(message)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")


class ExperimentRunner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Logger.log(f"[System] 使用设备: {self.device}")
        self._prepare_data()
        self._prepare_original_model()

    def _prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])
        test_set = datasets.CIFAR100(root=DATASET_DIR, train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_set, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    def _prepare_original_model(self):
        self.original_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
        self.original_model.to(self.device)
        self.original_model.eval()
        acc, _ = self.evaluate(self.original_model, key_factor=None, desc="Baseline")
        self.baseline_acc = acc
        Logger.log(f"[Check] 原始模型基准精度: {acc:.2f}% (Baseline)")

    def evaluate(self, model, key_factor, desc="Eval"):
        model.eval()
        correct = 0
        total = 0
        total_mse = 0.0
        mse_count = 0
        is_lock_model = isinstance(model, DynamicLOCKResNet56)
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if EVAL_LIMIT_BATCHES and i >= EVAL_LIMIT_BATCHES: break
                images, labels = images.to(self.device), labels.to(self.device)
                if is_lock_model:
                    outputs = model(images, runtime_key_factor=key_factor)
                else:
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                gt_outputs = self.original_model(images)
                batch_mse = nn.MSELoss()(outputs, gt_outputs).item()
                total_mse += batch_mse
                mse_count += 1
        avg_acc = 100 * correct / total
        avg_mse = total_mse / mse_count if mse_count > 0 else 0.0
        kf_str = f"{key_factor:.4f}" if key_factor else "N/A"
        Logger.log(f"   |-- {desc:<20} | Acc: {avg_acc:6.2f}% | MSE: {avg_mse:.6f} | Key: {kf_str}")
        return avg_acc, avg_mse

    def run_experiment_suite(self, experiment_list):
        Logger.log("\n" + "=" * 80)
        Logger.log(f"开始执行论文完整实验 (Final Fixed)")
        Logger.log("=" * 80)

        mk_A, salt_A = KeyManager.generate_master_key()
        mk_B, salt_B = KeyManager.generate_master_key()

        for exp in experiment_list:
            Logger.log(f"\n>>> 实验组: {exp['name']}")
            lock_model = DynamicLOCKResNet56(self.original_model, exp['points'])
            lock_model.to(self.device)

            uid_A = "user_A"
            uk_A = KeyManager.derive_user_key(mk_A, salt_A, uid_A)
            r_key_A = KeyManager.recover_runtime_factor(uk_A, salt_A, uid_A)
            Logger.log(f"    [Setup] User A 密钥因子: {r_key_A:.4f}")

            with torch.no_grad():
                # --- Point 0: 传统混淆 (Scale Conv Output) ---
                # 因为这是初始层，没有残差困扰，所以采用 Scale Conv + BN Stats 的方式
                if 0 in exp['points']:
                    alpha = np.random.uniform(exp['alpha_range'][0], exp['alpha_range'][1])
                    r_emb = 1.0 / (alpha * r_key_A)

                    lock_model.conv1.weight.data *= alpha
                    # 必须缩放 BN Stats，因为输入确实变小了
                    lock_model.bn1.weight.data *= alpha
                    lock_model.bn1.bias.data *= alpha
                    lock_model.bn1.running_mean.data *= alpha
                    lock_model.bn1.running_var.data *= (alpha ** 2)

                    node = lock_model.locks['0']
                    nn.init.dirac_(node.coupler.weight)
                    node.coupler.weight.data *= r_emb

                # --- Point 1, 2, 3: 混合策略 (Scale Next Layer Input) ---
                # 为了解决 ResNet Shortcut 问题，我们不缩放 Layer 输出，而是缩放 Next Layer 输入
                # 这样 信号 * 1/alpha (Coupler) * alpha (NextConv) = Normal
                # 这样 BN 就收到 Normal 信号，不需要修改 BN Stats

                # Point 1 -> Layer 2
                if 1 in exp['points']:
                    alpha = np.random.uniform(exp['alpha_range'][0], exp['alpha_range'][1])
                    r_emb = 1.0 / (alpha * r_key_A)

                    # 缩放 Layer 2 入口 Conv (Conv1 + Shortcut)
                    # 注意：不缩放 BN！
                    target_block = lock_model.layer2[0]
                    target_block.conv1.weight.data *= alpha

                    if target_block.downsample is not None:
                        # Downsample[0] is Conv
                        target_block.downsample[0].weight.data *= alpha

                    # Lock 1 负责缩小信号
                    node = lock_model.locks['1']
                    nn.init.dirac_(node.coupler.weight)
                    node.coupler.weight.data *= r_emb

                # Point 2 -> Layer 3
                if 2 in exp['points']:
                    alpha = np.random.uniform(exp['alpha_range'][0], exp['alpha_range'][1])
                    r_emb = 1.0 / (alpha * r_key_A)

                    target_block = lock_model.layer3[0]
                    target_block.conv1.weight.data *= alpha
                    if target_block.downsample is not None:
                        target_block.downsample[0].weight.data *= alpha

                    node = lock_model.locks['2']
                    nn.init.dirac_(node.coupler.weight)
                    node.coupler.weight.data *= r_emb

                # Point 3 -> FC
                if 3 in exp['points']:
                    alpha = np.random.uniform(exp['alpha_range'][0], exp['alpha_range'][1])
                    r_emb = 1.0 / (alpha * r_key_A)

                    # FC 直接缩放权重即可
                    lock_model.fc.weight.data *= alpha
                    if lock_model.fc.bias is not None:
                        lock_model.fc.bias.data *= alpha

                    node = lock_model.locks['3']
                    nn.init.dirac_(node.coupler.weight)
                    node.coupler.weight.data *= r_emb

            # 4. 评估
            self.evaluate(lock_model, key_factor=r_key_A, desc="Authorized")
            self.evaluate(lock_model, key_factor=1.0, desc="Attacker")

            uk_B = KeyManager.derive_user_key(mk_B, salt_B, "user_B")
            r_key_B = KeyManager.recover_runtime_factor(uk_B, salt_B, "user_B")
            self.evaluate(lock_model, key_factor=r_key_B, desc="Wrong UK (User B)")

            Logger.log(f"    [Done] 组 {exp['name']} 完成。")


if __name__ == "__main__":
    runner = ExperimentRunner()

    experiments = [
        # 深度对比
        {"name": "Exp 1-1 (单层)", "points": [0], "alpha_range": (0.1, 0.9)},
        {"name": "Exp 1-3 (全层)", "points": [0, 1, 2, 3], "alpha_range": (0.1, 0.9)},
        # 强度对比
        {"name": "Exp 2-1 (弱混淆)", "points": [0, 1, 2, 3], "alpha_range": (0.8, 0.95)},
        {"name": "Exp 2-2 (强混淆)", "points": [0, 1, 2, 3], "alpha_range": (0.01, 0.1)}
    ]

    runner.run_experiment_suite(experiments)
    Logger.log("\n所有实验结束。")