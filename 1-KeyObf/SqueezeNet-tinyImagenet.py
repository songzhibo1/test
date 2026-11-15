import torch
from pprint import pprint

# 第一次运行会从 GitHub 下载一个很小的 hubconf.py 脚本
# 存放在服务器的 ~/.cache/torch/hub 目录下
repo = "chenyaofo/pytorch-cifar-models"
models = torch.hub.list(repo, force_reload=True)

print(f"--- 共有 {len(models)} 个可用模型 ---")
pprint(models)