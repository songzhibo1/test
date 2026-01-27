#!/usr/bin/env python3
"""
将 CROWN 模型权重转换为 CROWN MPC C++ 所需的二进制格式

此脚本会生成:
1. weights/weights.dat - 权重文件 (按层顺序: W1, b1, W2, b2, ...)
2. images/X.bin - 输入图像文件
3. config.txt - 配置信息

用法:
    # 只需要指定模型文件名，自动解析所有参数
    python convert_for_crown_mpc.py --modelfile mnist_3layer_relu_1024_best
    python convert_for_crown_mpc.py --modelfile test_mnist_4layer_relu_20_best
    python convert_for_crown_mpc.py --modelfile mnist_4layer_relu_20
    python convert_for_crown_mpc.py --modelfile test_mnist_6layer_relu_20

支持的文件名格式:
    - mnist_3layer_relu_1024_best
    - mnist_3layer_relu_1024
    - test_mnist_4layer_relu_20_best
    - test_mnist_4layer_relu_20
    - cifar_5layer_relu_2048_best

输出目录结构:
    ./crown_mpc_data/{modelfile}/
    ├── weights/
    │   └── weights.dat
    ├── images/
    │   ├── 0.bin
    │   ├── 1.bin
    │   └── ...
    ├── labels.txt
    ├── config.txt
    └── README.md

注意:
    - 权重存储顺序: W1, b1, W2, b2, W3, b3, ...
    - 权重矩阵形状: (output_dim, input_dim) - 按行存储
    - 所有数据类型: float32
"""

import numpy as np
import gzip
import os
import argparse
import re

# ============================================================
# 配置
# ============================================================

# DEFAULT_CONFIG = {
#     'model': 'mnist',
#     'numlayer': 3,
#     'hidden': 1024,
#     'activation': 'relu',
#     'modeltype': 'best',
#     'data_dir': './data',
#     'models_dir': './models',
#     'output_dir': './crown_mpc_data',
#     'num_images': 10,
#     'eps': 0.1,
# }


DEFAULT_CONFIG = {
    'model': 'cifar',
    'numlayer': 5,
    'hidden': 100,
    'activation': 'relu',
    'modeltype': 'best',
    'data_dir': './cifar-10-batches-bin',
    'models_dir': './models',
    'output_dir': './crown_mpc_data',
    'num_images': 10,
    'eps': 0.1,
}

# ============================================================
# 从文件名解析参数
# ============================================================

def parse_modelfile(modelfile):
    """
    从模型文件名解析所有参数

    支持的格式:
        - mnist_3layer_relu_1024_best
        - mnist_3layer_relu_1024
        - test_mnist_4layer_relu_20_best
        - test_mnist_4layer_relu_20
        - cifar_5layer_relu_2048_distill

    返回:
        dict: {
            'model': 'mnist' or 'cifar',
            'numlayer': int,
            'activation': str,
            'hidden': int,
            'modeltype': str or None
        }
    """
    result = {
        'model': 'mnist',
        'numlayer': 3,
        'activation': 'relu',
        'hidden': 1024,
        'modeltype': None,  # None 表示文件名中没有 modeltype
    }

    # 移除路径，只保留文件名
    filename = os.path.basename(modelfile)

    # 判断数据集类型
    if 'cifar' in filename.lower():
        result['model'] = 'cifar'
    else:
        result['model'] = 'mnist'

    # 解析层数: 匹配 "3layer" 或 "4layer" 等
    layer_match = re.search(r'(\d+)layer', filename)
    if layer_match:
        result['numlayer'] = int(layer_match.group(1))

    # 解析激活函数
    activations = ['relu', 'tanh', 'sigmoid', 'arctan', 'elu', 'softplus']
    for act in activations:
        if act in filename.lower():
            result['activation'] = act
            break

    # 解析隐藏单元数: 在激活函数之后的数字
    # 例如: mnist_3layer_relu_1024_best -> 1024
    # 使用正则表达式匹配 activation_数字 的模式
    hidden_match = re.search(r'(?:relu|tanh|sigmoid|arctan|elu|softplus)_(\d+)', filename.lower())
    if hidden_match:
        result['hidden'] = int(hidden_match.group(1))

    # 解析模型类型 (best, adv_retrain, distill, vanilla)
    modeltypes = ['best', 'adv_retrain', 'distill', 'vanilla']
    for mt in modeltypes:
        if filename.endswith(mt) or f'_{mt}' in filename:
            result['modeltype'] = mt
            break

    return result


# ============================================================
# 生成输出文件夹名称
# ============================================================

def generate_output_folder_name(config):
    """
    根据参数自动生成输出文件夹名称

    格式: {model}_{numlayer}layer_{activation}_{hidden}_{modeltype}
    示例: mnist_3layer_relu_1024_best
    """
    folder_name = f"{config['model']}_{config['numlayer']}layer_{config['activation']}_{config['hidden']}"

    # 添加 modeltype 后缀
    if config['modeltype'] and config['modeltype'] != 'vanilla':
        folder_name += f"_{config['modeltype']}"

    return folder_name


def get_full_output_path(config):
    """
    获取完整的输出路径

    返回: {output_dir}/{generated_folder_name}
    示例: ./crown_mpc_data/mnist_3layer_relu_1024_best
    """
    folder_name = generate_output_folder_name(config)
    full_path = os.path.join(config['output_dir'], folder_name)
    return full_path


# ============================================================
# MNIST 数据加载
# ============================================================

def load_mnist_images(data_dir, num_images=10):
    """加载 MNIST 测试图像"""
    images_file = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    labels_file = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    if not os.path.exists(images_file):
        print(f"Error: MNIST data not found at {images_file}")
        print("Please download MNIST data first or check --data_dir path")
        return None, None

    # 加载图像
    with gzip.open(images_file) as f:
        f.read(16)
        buf = f.read(10000 * 28 * 28)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = (images / 255) - 0.5  # 归一化到 [-0.5, 0.5]
        images = images.reshape(10000, 784)

    # 加载标签
    with gzip.open(labels_file) as f:
        f.read(8)
        labels = np.frombuffer(f.read(10000), dtype=np.uint8)

    return images[:num_images], labels[:num_images]


# ============================================================
# CIFAR 数据加载
# ============================================================

def load_cifar_images(data_dir, num_images=10):
    """加载 CIFAR-10 测试图像"""
    test_file_bin = os.path.join(data_dir, "cifar-10-batches-bin", "test_batch.bin")

    if os.path.exists(test_file_bin):
        print(f"  Loading CIFAR from binary format: {test_file_bin}")
        with open(test_file_bin, "rb") as f:
            data = f.read()

        size = 32 * 32 * 3 + 1
        images = []
        labels = []

        num_to_load = min(num_images, 10000)
        for i in range(num_to_load):
            arr = np.frombuffer(data[i * size:(i + 1) * size], dtype=np.uint8)
            labels.append(arr[0])

            # 关键修复：与原始 setup_cifar.py 保持一致
            img = arr[1:].reshape((3, 32, 32)).transpose((1, 2, 0))  # (3,32,32) -> (32,32,3)
            img = (img.astype(np.float32) / 255) - 0.5

            # Flatten 成与 Keras Flatten 一致的顺序
            images.append(img.flatten())  # (32,32,3) flatten = RGBRGBRGB... 顺序

        return np.array(images), np.array(labels)

# ============================================================
# 模型权重加载
# ============================================================

def load_crown_weights(model_path, numlayer, hidden, model_type='mnist', activation='relu'):
    """
    加载 CROWN Keras 模型权重

    返回:
        weights: list of numpy arrays, 每个形状为 (output_dim, input_dim)
        biases: list of numpy arrays, 每个形状为 (output_dim,)
        layer_dims: list of ints, [input_dim, hidden1, hidden2, ..., output_dim]
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Activation, Flatten
    except ImportError:
        print("Error: TensorFlow not installed.")
        print("Please install with: pip install tensorflow")
        return None, None, None

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None, None, None

    # 确定输入维度
    if model_type == 'mnist':
        input_dim = 784  # 28x28
        image_size = 28
        image_channel = 1
    else:  # cifar
        input_dim = 3072  # 32x32x3
        image_size = 32
        image_channel = 3

    output_dim = 10  # 10 classes

    # 构建模型结构
    model = Sequential()
    model.add(Flatten(input_shape=(image_size, image_size, image_channel)))

    for _ in range(numlayer - 1):
        model.add(Dense(hidden))
        model.add(Activation(activation))

    model.add(Dense(output_dim))

    # 加载权重
    print(f"Loading model: {model_path}")
    model.load_weights(model_path)

    # 提取权重
    weights = []
    biases = []
    layer_dims = [input_dim]

    for layer in model.layers:
        w = layer.get_weights()
        if len(w) == 2:  # Dense 层
            W, b = w
            # W 的形状是 (input_dim, output_dim)，需要转置为 (output_dim, input_dim)
            W_transposed = W.T
            weights.append(W_transposed.astype(np.float32))
            biases.append(b.astype(np.float32))
            layer_dims.append(W.shape[1])
            print(f"  Layer: W={W.shape} -> {W_transposed.shape}, b={b.shape}")

    return weights, biases, layer_dims


# ============================================================
# 文件保存
# ============================================================

def save_weights_binary(weights, biases, output_path):
    """
    保存权重为二进制格式

    格式: W1, b1, W2, b2, W3, b3, ... (按顺序连续存储)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        total_params = 0
        for i, (W, b) in enumerate(zip(weights, biases)):
            # 保存权重矩阵 (已经是 output_dim x input_dim 形状)
            W_flat = W.flatten()
            W_flat.tofile(f)
            total_params += W_flat.size
            print(f"  W{i + 1}: shape={W.shape}, size={W_flat.size}")

            # 保存偏置
            b.tofile(f)
            total_params += b.size
            print(f"  b{i + 1}: shape={b.shape}, size={b.size}")

    file_size = os.path.getsize(output_path)
    print(f"\nWeights saved to: {output_path}")
    print(f"  Total parameters: {total_params}")
    print(f"  File size: {file_size} bytes ({file_size / 1024:.2f} KB)")


def save_images_binary(images, labels, output_dir):
    """保存图像为二进制格式"""
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    for i, (img, label) in enumerate(zip(images, labels)):
        filepath = os.path.join(images_dir, f'{i}.bin')
        img.astype(np.float32).tofile(filepath)
        print(f"  Image {i}: label={label}, saved to {filepath}")

    # 保存标签
    labels_path = os.path.join(output_dir, 'labels.txt')
    np.savetxt(labels_path, labels, fmt='%d')
    print(f"  Labels saved to: {labels_path}")


def save_config(config, layer_dims, output_dir, folder_name):
    """保存配置信息"""
    config_path = os.path.join(output_dir, 'config.txt')

    with open(config_path, 'w') as f:
        f.write("# CROWN MPC Configuration\n")
        f.write(f"# Model file: {config.get('modelfile', folder_name)}\n\n")
        f.write(f"num_layers={len(layer_dims) - 1}\n")
        f.write(f"layer_dims={','.join(map(str, layer_dims))}\n")
        f.write(f"weights_file=weights/weights.dat\n")
        f.write(f"eps={config['eps']}\n")
        f.write(f"\n# Parsed parameters\n")
        f.write(f"model={config['model']}\n")
        f.write(f"numlayer={config['numlayer']}\n")
        f.write(f"hidden={config['hidden']}\n")
        f.write(f"activation={config['activation']}\n")
        if config.get('modeltype'):
            f.write(f"modeltype={config['modeltype']}\n")

    print(f"\nConfig saved to: {config_path}")


# ============================================================
# 生成 C++ 使用示例
# ============================================================

def generate_usage_example(config, layer_dims, output_dir, folder_name):
    """生成 C++ 使用示例"""
    readme_path = os.path.join(output_dir, 'README.md')

    dims_str = " -> ".join(map(str, layer_dims))

    content = f"""# CROWN MPC 数据文件

## 模型信息
- **模型文件**: `{config.get('modelfile', folder_name)}`
- **输出文件夹**: `{folder_name}`
- **解析的参数**: 
  - model: {config['model']}
  - numlayer: {config['numlayer']}
  - hidden: {config['hidden']}
  - activation: {config['activation']}
  - modeltype: {config.get('modeltype', 'None')}

## 网络结构
- 层数: {len(layer_dims) - 1}
- 维度: {dims_str}
- 激活函数: {config['activation']}

## 文件结构
```
{folder_name}/
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
- 大小: {layer_dims[0]} 个值
- 数值范围: [-0.5, 0.5]

## C++ 使用示例

```bash
# 运行 MPC 程序
./crown_mpc \\
    --weights={folder_name}/weights/weights.dat \\
    --input={folder_name}/images/0.bin \\
    --eps={config['eps']} \\
    --true_label=7 \\
    --target_label=1
```

## C++ 配置代码

```cpp
NetworkConfig config;
config.num_layers = {len(layer_dims) - 1};
config.layer_dims = {{{', '.join(map(str, layer_dims))}}};
config.weights_file = "{folder_name}/weights/weights.dat";
config.input_file = "{folder_name}/images/0.bin";
config.eps = {config['eps']}f;
```

## 验证数据

```python
import numpy as np

# 验证图像
img = np.fromfile('{folder_name}/images/0.bin', dtype=np.float32)
print(f"Image shape: {{img.shape}}, range: [{{img.min():.3f}}, {{img.max():.3f}}]")

# 验证权重
weights = np.fromfile('{folder_name}/weights/weights.dat', dtype=np.float32)
print(f"Total weights: {{weights.shape[0]}}")
```

## 标签对照
标签文件 `labels.txt` 包含每张图像的真实标签。
"""

    with open(readme_path, 'w') as f:
        f.write(content)

    print(f"README saved to: {readme_path}")


# ============================================================
# 验证函数
# ============================================================

def verify_weights(weights_path, weights, biases):
    """验证保存的权重文件"""
    print("\nVerifying weights file...")

    data = np.fromfile(weights_path, dtype=np.float32)

    offset = 0
    all_match = True

    for i, (W, b) in enumerate(zip(weights, biases)):
        W_size = W.size
        b_size = b.size

        W_loaded = data[offset:offset + W_size].reshape(W.shape)
        offset += W_size

        b_loaded = data[offset:offset + b_size]
        offset += b_size

        W_match = np.allclose(W, W_loaded)
        b_match = np.allclose(b, b_loaded)

        print(f"  Layer {i + 1}: W match={W_match}, b match={b_match}")

        if not W_match or not b_match:
            all_match = False

    if all_match:
        print("  ✓ All weights verified successfully!")
    else:
        print("  ✗ Weight verification failed!")

    return all_match


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert CROWN model for MPC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 单个模型文件
    python convert_for_crown_mpc.py --modelfile mnist_3layer_relu_1024_best

    # 多个模型文件
    python convert_for_crown_mpc.py --modelfile mnist_3layer_relu_1024_best test_mnist_4layer_relu_20_best

    # 使用默认模型列表
    python convert_for_crown_mpc.py

支持的文件名格式:
    - mnist_3layer_relu_1024_best      -> model=mnist, numlayer=3, hidden=1024, modeltype=best
    - test_mnist_4layer_relu_20_best   -> model=mnist, numlayer=4, hidden=20, modeltype=best
    - mnist_4layer_relu_20             -> model=mnist, numlayer=4, hidden=20, modeltype=None
    - cifar_5layer_relu_2048_distill   -> model=cifar, numlayer=5, hidden=2048, modeltype=distill
        """
    )

    # 默认模型文件列表
    default_modelfiles = [
        # 'test_mnist_5layer_relu_20_best',
        # 'test_mnist_6layer_relu_20_best',
        # 'test_mnist_8layer_relu_20_best',
        'vnncomp_mnist_3layer_relu_256_best',
        'vnncomp_mnist_5layer_relu_256_best',
        'vnncomp_mnist_7layer_relu_256_best',
        # 'cifar_6layer_relu_2048_best',
        'eran_cifar_5layer_relu_100_best',
        'eran_cifar_7layer_relu_100_best',
        'eran_cifar_10layer_relu_200_best',
    ]

    parser.add_argument('--modelfile', nargs='+', type=str,
                        default=default_modelfiles,
                        help='模型文件名列表，支持多个文件 (default: %(default)s)')
    parser.add_argument('--data_dir', default='../../data',
                        help='MNIST/CIFAR 数据目录 (default: ../../data)')
    parser.add_argument('--models_dir', default='../../models',
                        help='CROWN 模型目录 (default: ../../models)')
    parser.add_argument('--output_dir', default='./crown_mpc_data',
                        help='输出根目录 (default: ./crown_mpc_data)')
    parser.add_argument('--num_images', default=100, type=int,
                        help='转换的图像数量 (default: 10)')
    parser.add_argument('--eps', default=0.1, type=float,
                        help='扰动半径 epsilon (default: 0.1)')

    args = parser.parse_args()

    # 获取模型文件列表
    modelfile_list = args.modelfile

    print("=" * 60)
    print("CROWN Model Converter for MPC")
    print("=" * 60)
    print(f"\n待处理的模型文件 ({len(modelfile_list)} 个):")
    for i, mf in enumerate(modelfile_list):
        print(f"  [{i + 1}] {mf}")
    print("=" * 60)

    # 记录处理结果
    success_list = []
    failed_list = []

    # 遍历处理每个模型文件
    for idx, modelfile in enumerate(modelfile_list):
        print(f"\n{'#' * 60}")
        print(f"# 处理模型 [{idx + 1}/{len(modelfile_list)}]: {modelfile}")
        print(f"{'#' * 60}")

        try:
            result = process_single_model(
                modelfile=modelfile,
                data_dir=args.data_dir,
                models_dir=args.models_dir,
                output_dir=args.output_dir,
                num_images=args.num_images,
                eps=args.eps
            )
            if result:
                success_list.append(modelfile)
            else:
                failed_list.append(modelfile)
        except Exception as e:
            print(f"Error processing {modelfile}: {e}")
            failed_list.append(modelfile)

    # 打印最终汇总
    print("\n" + "=" * 60)
    print("处理完成汇总")
    print("=" * 60)
    print(f"\n成功: {len(success_list)} 个")
    for mf in success_list:
        print(f"  ✓ {mf}")

    if failed_list:
        print(f"\n失败: {len(failed_list)} 个")
        for mf in failed_list:
            print(f"  ✗ {mf}")

    print("=" * 60)


def process_single_model(modelfile, data_dir, models_dir, output_dir, num_images, eps):
    """
    处理单个模型文件 (已修改：自动根据模型类型锁定数据路径)
    """
    # ============================================================
    # 从文件名解析参数
    # ============================================================
    parsed = parse_modelfile(modelfile)

    # ============================================================
    # [修改点] 根据模型类型，在这里写死两个不同的数据目录
    # ============================================================
    if parsed['model'] == 'mnist':
        # MNIST: 截图显示文件在 crown/data
        # 这里的路径指向 data 文件夹
        real_data_dir = '../../data'
    else:
        # CIFAR: 截图显示文件在 crown/cifar-10-batches-bin
        # 注意：下面的 load_cifar_images 函数会自动拼接 "/cifar-10-batches-bin"
        # 所以这里我们只需要指向它的父目录 (即 crown 根目录 ../../)
        real_data_dir = '../../'

    # 合并配置
    config = {
        'modelfile': modelfile,
        'model': parsed['model'],
        'numlayer': parsed['numlayer'],
        'activation': parsed['activation'],
        'hidden': parsed['hidden'],
        'modeltype': parsed['modeltype'],
        'data_dir': real_data_dir,  # <--- 这里使用了上面定义好的路径
        'models_dir': models_dir,
        'output_dir': output_dir,
        'num_images': num_images,
        'eps': eps,
    }

    # 输出文件夹名 = 输入的模型文件名
    folder_name = modelfile
    full_output_path = os.path.join(config['output_dir'], folder_name)
    model_path = os.path.join(config['models_dir'], modelfile)

    print(f"\n模型文件名: {modelfile}")
    print(f"\n自动解析的参数:")
    print(f"  model      = {config['model']}")
    print(f"  numlayer   = {config['numlayer']}")
    print(f"  hidden     = {config['hidden']}")
    print(f"  activation = {config['activation']}")
    print(f"  modeltype  = {config['modeltype']}")
    print(f"\n模型文件路径: {model_path}")
    print(f"数据文件路径: {config['data_dir']} (自动设定)")
    print(f"输出文件夹: {full_output_path}")

    # 创建输出目录
    os.makedirs(full_output_path, exist_ok=True)

    # 1. 加载图像
    print("\n[Step 1] Loading images...")
    if config['model'] == 'mnist':
        # MNIST 加载器直接读取目录下的 .gz 文件
        images, labels = load_mnist_images(config['data_dir'], config['num_images'])
    else:
        # CIFAR 加载器会读取 config['data_dir'] + "/cifar-10-batches-bin/test_batch.bin"
        # 所以上面 real_data_dir 设置为 ../../ 是正确的
        images, labels = load_cifar_images(config['data_dir'], config['num_images'])

    if images is None:
        print("Error: Failed to load images. Please check paths in script.")
        return False
    print(f"  Loaded {len(images)} images")

    # 2. 保存图像
    print("\n[Step 2] Saving images...")
    save_images_binary(images, labels, full_output_path)

    # 3. 加载权重
    print("\n[Step 3] Loading model weights...")
    weights, biases, layer_dims = load_crown_weights(
        model_path,
        config['numlayer'],
        config['hidden'],
        config['model'],
        config['activation']
    )

    if weights is None:
        print("Error: Failed to load model weights")
        return False

    # 4. 保存权重
    print("\n[Step 4] Saving weights...")
    weights_path = os.path.join(full_output_path, 'weights', 'weights.dat')
    save_weights_binary(weights, biases, weights_path)

    # 5. 保存配置
    print("\n[Step 5] Saving configuration...")
    save_config(config, layer_dims, full_output_path, folder_name)

    # 6. 生成使用说明
    print("\n[Step 6] Generating documentation...")
    generate_usage_example(config, layer_dims, full_output_path, folder_name)

    # 7. 验证
    print("\n[Step 7] Verifying...")
    verify_weights(weights_path, weights, biases)

    # 打印结果
    print(f"\n✓ Model '{modelfile}' conversion complete!")
    print(f"  输出目录: {full_output_path}/")

    return True
# def process_single_model(modelfile, data_dir, models_dir, output_dir, num_images, eps):
#     """
#     处理单个模型文件
#
#     返回:
#         bool: 成功返回 True，失败返回 False
#     """
#     # ============================================================
#     # 从文件名解析参数
#     # ============================================================
#     parsed = parse_modelfile(modelfile)
#
#     # 合并配置
#     config = {
#         'modelfile': modelfile,
#         'model': parsed['model'],
#         'numlayer': parsed['numlayer'],
#         'activation': parsed['activation'],
#         'hidden': parsed['hidden'],
#         'modeltype': parsed['modeltype'],
#         'data_dir': data_dir,
#         'models_dir': models_dir,
#         'output_dir': output_dir,
#         'num_images': num_images,
#         'eps': eps,
#     }
#
#     # 输出文件夹名 = 输入的模型文件名
#     folder_name = modelfile
#     full_output_path = os.path.join(config['output_dir'], folder_name)
#     model_path = os.path.join(config['models_dir'], modelfile)
#
#     print(f"\n模型文件名: {modelfile}")
#     print(f"\n自动解析的参数:")
#     print(f"  model      = {config['model']}")
#     print(f"  numlayer   = {config['numlayer']}")
#     print(f"  hidden     = {config['hidden']}")
#     print(f"  activation = {config['activation']}")
#     print(f"  modeltype  = {config['modeltype']}")
#     print(f"\n模型文件路径: {model_path}")
#     print(f"输出文件夹: {full_output_path}")
#
#     # 创建输出目录
#     os.makedirs(full_output_path, exist_ok=True)
#
#     # 1. 加载图像
#     print("\n[Step 1] Loading images...")
#     if config['model'] == 'mnist':
#         images, labels = load_mnist_images(config['data_dir'], config['num_images'])
#     else:
#         images, labels = load_cifar_images(config['data_dir'], config['num_images'])
#
#     if images is None:
#         print("Error: Failed to load images. Please check --data_dir path.")
#         return False
#     print(f"  Loaded {len(images)} images")
#
#     # 2. 保存图像
#     print("\n[Step 2] Saving images...")
#     save_images_binary(images, labels, full_output_path)
#
#     # 3. 加载权重
#     print("\n[Step 3] Loading model weights...")
#     weights, biases, layer_dims = load_crown_weights(
#         model_path,
#         config['numlayer'],
#         config['hidden'],
#         config['model'],
#         config['activation']
#     )
#
#     if weights is None:
#         print("Error: Failed to load model weights")
#         return False
#
#     # 4. 保存权重
#     print("\n[Step 4] Saving weights...")
#     weights_path = os.path.join(full_output_path, 'weights', 'weights.dat')
#     save_weights_binary(weights, biases, weights_path)
#
#     # 5. 保存配置
#     print("\n[Step 5] Saving configuration...")
#     save_config(config, layer_dims, full_output_path, folder_name)
#
#     # 6. 生成使用说明
#     print("\n[Step 6] Generating documentation...")
#     generate_usage_example(config, layer_dims, full_output_path, folder_name)
#
#     # 7. 验证
#     print("\n[Step 7] Verifying...")
#     verify_weights(weights_path, weights, biases)
#
#     # 打印结果
#     print(f"\n✓ Model '{modelfile}' conversion complete!")
#     print(f"  输出目录: {full_output_path}/")
#
#     return True


if __name__ == "__main__":
    main()