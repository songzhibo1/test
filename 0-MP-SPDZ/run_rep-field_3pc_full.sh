#!/bin/bash

# ==============================================================================
# 批量执行脚本: 全 MNIST 模型 rep-field 协议实证测试
#
# rep-field = 3PC honest-majority, SEMI-HONEST security (Replicated Secret Sharing)
# 域:         素域 (prime field)
# 预处理:     几乎为 0 ! 只需共享 PRNG 种子, 无需 OT/HE
# 在线:       直接乘法 + resharing (1 轮, 1 elem/方, 比 semi 更少通信)
# 关键对比:   与 semi 相同的安全模型和域, 差别是方数 (2 -> 3)
#            预期: offline 接近 0, online 可能是所有协议里最快的
#
# 注意:
#   - 需要 3 方运行 (自动在 localhost 启动 3 个 player 进程)
#   - 需要已编译 replicated-field-party.x  (make -j4 replicated-field-party.x)
#   - Fake-Offline.x 将自动使用 N_PARTIES=3
#   - crown_prepare_data.py 将自动为 P2 生成空输入文件
#
# 结果保存在: crown-results/rep-field_3pc/<variant>/<model>/eps_<eps>/
# ==============================================================================

PROTOCOL="rep-field"

SCRIPTS=(
    "./run_crown_elemwise.sh"
    "./run_crown_batchrelu.sh"
    "./run_crown_batchsplit.sh"
    "./run_crown_navieopt.sh"
    "./run_crown_naive.sh"
)

TEST_CASES=(
    "mnist_2layer_20 0.04500 0 7 4"
    "mnist_3layer_20 0.03000 0 7 6"
    "mnist_3layer_256 0.01500 0 7 3"
    "mnist_5layer_256 0.01500 0 7 8"
    "mnist_7layer_256 0.01500 0 7 3"
)

echo "=============================================================================="
echo "开始 rep-field 协议 (3PC semi-honest, replicated) 全 MNIST 实证测试..."
echo "策略: 从小到大运行, 自动清理 Player-Data/*.sch 以节省磁盘."
echo "=============================================================================="

# Sanity check: 二进制存在
if [ ! -f "./replicated-field-party.x" ]; then
    echo "!! 警告: replicated-field-party.x 未编译."
    echo "   请先执行: make -j4 replicated-field-party.x"
    echo "   继续尝试运行 (可能失败)..."
fi

for case in "${TEST_CASES[@]}"; do
    read -r MODEL EPS ID TRUE TARGET <<< "$case"

    echo ""
    echo "##############################################################################"
    echo "[CASE START] Protocol: $PROTOCOL | Model: $MODEL | EPS: $EPS"
    echo "##############################################################################"

    for SCRIPT in "${SCRIPTS[@]}"; do
        if [ ! -f "$SCRIPT" ]; then
            echo "!! 跳过 $SCRIPT (文件不存在)"
            continue
        fi

        echo "--------------------------------------------------------"
        echo ">>> Variant: $SCRIPT | Protocol: $PROTOCOL"
        echo "--------------------------------------------------------"

        chmod +x "$SCRIPT"

        CROWN_EPS=$EPS \
        CROWN_IMAGE_ID=$ID \
        CROWN_TRUE_LABEL=$TRUE \
        CROWN_TARGET_LABEL=$TARGET \
        $SCRIPT $PROTOCOL "$MODEL"

        echo "清理 Player-Data/*.sch ..."
        rm -rf Player-Data/*.sch

        echo "准备进入下一个变体."
        sleep 3
    done
done

echo ""
echo "=============================================================================="
echo "rep-field 协议全部 MNIST 测试完成!"
echo "结果目录: crown-results/rep-field_3pc/"
echo "=============================================================================="
