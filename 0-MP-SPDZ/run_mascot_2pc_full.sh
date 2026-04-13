#!/bin/bash

# ==============================================================================
# 批量执行脚本: 全 MNIST 模型 mascot 协议实证测试
#
# mascot = 2PC dishonest-majority, MALICIOUS security (SPDZ-style OT+MAC)
# 域:     素域 (prime field)
# 预处理: OT extension + sacrifice + MAC
# 在线:   Beaver triples + MAC verification
#
# 结果保存在: crown-results/mascot_2pc/<variant>/<model>/eps_<eps>/
#
# 注意:
#   - mascot 的 mnist_7layer_256 可能较慢 (恶意安全 offline 比 semi 贵 2-3x)
#   - 若磁盘紧张,建议先跑 2-5 层模型
# ==============================================================================

PROTOCOL="mascot"

# 5 种 MPC 变体
SCRIPTS=(
    "./run_crown_elemwise.sh"
    "./run_crown_batchrelu.sh"
    "./run_crown_batchsplit.sh"
    "./run_crown_navieopt.sh"
    "./run_crown_naive.sh"
)

# MNIST 测试用例 (从小到大)
TEST_CASES=(
    "mnist_2layer_20 0.04500 0 7 4"
    "mnist_3layer_20 0.03000 0 7 6"
    "mnist_3layer_256 0.01500 0 7 3"
    "mnist_5layer_256 0.01500 0 7 8"
    "mnist_7layer_256 0.01500 0 7 3"
)

echo "=============================================================================="
echo "开始 mascot 协议 (2PC malicious, field) 全 MNIST 实证测试..."
echo "策略: 从小到大运行, 自动清理 Player-Data/*.sch 以节省磁盘."
echo "=============================================================================="

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

        # 清理预处理缓存
        echo "清理 Player-Data/*.sch ..."
        rm -rf Player-Data/*.sch

        echo "准备进入下一个变体."
        sleep 3
    done
done

echo ""
echo "=============================================================================="
echo "mascot 协议全部 MNIST 测试完成!"
echo "结果目录: crown-results/mascot_2pc/"
echo "=============================================================================="
