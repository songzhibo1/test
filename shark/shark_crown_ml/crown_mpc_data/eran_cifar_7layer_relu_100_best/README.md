# CROWN MPC 数据文件

## 模型信息
- **模型文件**: `eran_cifar_7layer_relu_100_best`
- **输出文件夹**: `eran_cifar_7layer_relu_100_best`
- **解析的参数**: 
  - model: cifar
  - numlayer: 7
  - hidden: 100
  - activation: relu
  - modeltype: best

## 网络结构
- 层数: 7
- 维度: 3072 -> 100 -> 100 -> 100 -> 100 -> 100 -> 100 -> 10
- 激活函数: relu

## 文件结构
```
eran_cifar_7layer_relu_100_best/
├── weights/
│   └── weights.dat      # 模型权重 (W1,b1,W2,b2,...)
├── images/
│   ├── 0.bin            # 测试图像
│   ├── 1.bin
│   └── ...
├── labels.txt           # 图像标签
├── config.txt           # 配置信息
└── README.md            # 本文件
```

## 文件格式说明

### weights/weights.dat
- 格式: 二进制 float32
- 存储顺序: W1, b1, W2, b2, W3, b3, ...
- W 矩阵形状: (output_dim, input_dim) 按行存储

### images/X.bin
- 格式: 二进制 float32
- 大小: 3072 个值
- 数值范围: [-0.5, 0.5]

## C++ 使用示例

```bash
# 运行 MPC 程序
./crown_mpc \
    --weights=eran_cifar_7layer_relu_100_best/weights/weights.dat \
    --input=eran_cifar_7layer_relu_100_best/images/0.bin \
    --eps=0.1 \
    --true_label=7 \
    --target_label=1
```

## C++ 配置代码

```cpp
NetworkConfig config;
config.num_layers = 7;
config.layer_dims = {3072, 100, 100, 100, 100, 100, 100, 10};
config.weights_file = "eran_cifar_7layer_relu_100_best/weights/weights.dat";
config.input_file = "eran_cifar_7layer_relu_100_best/images/0.bin";
config.eps = 0.1f;
```

## 验证数据

```python
import numpy as np

# 验证图像
img = np.fromfile('eran_cifar_7layer_relu_100_best/images/0.bin', dtype=np.float32)
print(f"Image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")

# 验证权重
weights = np.fromfile('eran_cifar_7layer_relu_100_best/weights/weights.dat', dtype=np.float32)
print(f"Total weights: {weights.shape[0]}")
```

## 标签对照
标签文件 `labels.txt` 包含每张图像的真实标签。
