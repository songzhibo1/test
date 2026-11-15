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

# =========================================================================
# 0. 环境配置
# =========================================================================
ssl._create_default_https_context = ssl._create_unverified_context
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "datasets")
os.makedirs(DATASET_DIR, exist_ok=True)

# 缩放因子：使用更极端的范围增强扰动
ALPHA_RANGE = (0.05, 0.15)


# =========================================================================
# 1. 强化型密钥管理器 (KeyManager)
# =========================================================================
class KeyManager:
    def __init__(self):
        self.master_key = b'DynaMO_Deep_Secret_2026'

    def get_params(self, uid, layer_id):
        if uid is None:
            return None  # 代表无密钥状态

        msg = f"{uid}_{layer_id}".encode()
        h = hmac.new(self.master_key, msg, hashlib.sha256).digest()
        seed = int.from_bytes(h[:4], 'big')
        rng = np.random.RandomState(seed)

        # 派生 alpha
        alpha = rng.uniform(ALPHA_RANGE[0], ALPHA_RANGE[1])
        # 派生验证指纹 (Fingerprint)，用于校验密钥是否匹配
        fingerprint = rng.uniform(0, 1)
        return {"alpha": alpha, "fp": fingerprint}


# =========================================================================
# 2. 最终版 DynaMO 混淆封装
# =========================================================================
class DynaMO_ResNet_Ultimate(nn.Module):
    def __init__(self, base_model, key_manager, admin_uid):
        super().__init__()
        self.backbone = copy.deepcopy(base_model)
        self.km = key_manager
        self.admin_uid = admin_uid

        print(f"\n[DynaMO] 正在应用非线性塌陷混淆 (Alpha: {ALPHA_RANGE})...")

        for name, m in self.backbone.named_modules():
            if m.__class__.__name__ == 'BasicBlock':
                layer_id = f"block_{name.replace('.', '_')}"

                # 获取 Admin 初始化的参数
                params = self.km.get_params(self.admin_uid, layer_id)
                a_admin = params['alpha']
                fp_admin = params['fp']

                with torch.no_grad():
                    # Lemma 4.1: 静态缩放 W 和 BN
                    m.conv1.weight.data *= a_admin
                    m.bn1.weight.data *= a_admin
                    m.bn1.bias.data *= a_admin
                    m.bn1.running_mean.data *= a_admin
                    m.bn1.running_var.data *= (a_admin ** 2)

                # 存储该层的正确指纹（用于在 forward 中判断密钥是否正确）
                self.register_buffer(f"{layer_id}_target_fp", torch.tensor(fp_admin))
                self.register_buffer(f"{layer_id}_target_a", torch.tensor(a_admin))

                self._patch_block(m, layer_id)

    def _patch_block(self, block, layer_id):
        def new_forward(x):
            identity = x

            # 1. 残差分支计算
            out = block.conv1(x)
            out = block.bn1(out)
            out = block.relu(out)

            # 2. GDSOGU 增强校验逻辑 (Theorem 4.2)
            current_uid = getattr(self, 'current_uid', None)
            user_params = self.km.get_params(current_uid, layer_id)

            target_fp = getattr(self, f"{layer_id}_target_fp")
            target_a = getattr(self, f"{layer_id}_target_a")

            if user_params is not None and abs(user_params['fp'] - target_fp.item()) < 1e-6:
                # 密钥完全匹配：执行精确修正
                correction = 1.0 / target_a
            else:
                # 密钥不匹配或无密钥：引入非线性塌陷
                # 不仅不修正，反而施加一个极小的扰动，使特征图在相加时产生数值截断
                correction = 0.001

            out = out * correction

            # 3. 后续层与残差合并
            out = block.conv2(out)
            out = block.bn2(out)

            if block.downsample is not None:
                identity = block.downsample(x)

            # 如果 correction 极小，卷积分支被淹没；如果 correction 错误，特征分布被破坏
            out += identity
            return block.relu(out)

        block.forward = new_forward

    def forward(self, x, uid=None):
        self.current_uid = uid
        return self.backbone(x)


# =========================================================================
# 3. 运行评估
# =========================================================================
def main():
    km = KeyManager()
    admin_uid = "TJ-2026-ROOT"

    print(">>> 正在加载 ResNet56...")
    raw_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True).to(DEVICE)

    model = DynaMO_ResNet_Ultimate(raw_model, km, admin_uid).to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_loader = DataLoader(datasets.CIFAR100(root=DATASET_DIR, train=False, download=True, transform=transform),
                             batch_size=128, shuffle=False)

    def evaluate(uid, label):
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs, uid=uid)
                correct += (outputs.argmax(1) == lbls).sum().item()
                total += lbls.size(0)
        print(f"   [{label}] Accuracy: {100 * correct / total:.2f}%")

    print("\n" + "=" * 50)
    print("DynaMO 最终强化版测试")
    print("=" * 50)
    evaluate(admin_uid, "授权用户 (Correct Key)")
    evaluate("WRONG_KEY_888", "错误密钥 (Wrong Key)")
    evaluate(None, "无密钥访问 (No Key)")


if __name__ == "__main__":
    main()