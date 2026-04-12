#!/bin/bash

# ==============================================================================
# 批量执行脚本: 全 MNIST 模型 mal-rep-field 协议实证测试
#
# mal-rep-field = 3PC honest-majority, MALICIOUS security
# 域:             素域 (prime field)
# 预处理:         少量 sacrifice 比特, 但远比 2PC 恶意协议 (mascot) 便宜
# 在线:           3PC 复制分享乘法 + post-sacrifice 或 hash-based 验证
# 关键对比:       与 rep-field 相同方数/域, 差别是安全模型 (半诚实 -> 恶意)
#                与 mascot 相同安全模型, 差别是方数 (2 -> 3)
#                展示 "3PC honest-majority + malicious" 组合的性价比
#
# 注意:
#   - 需要 3 方运行 (自动在 localhost 启动 3 个 player 进程)
#   - 需要已编译 malicious-rep-field-party.x  (make -j4 malicious-rep-field-party.x)
#   - mnist_7layer_256 可能较慢, 建议最后跑
#
# 结果保存在: crown-results/mal-rep-field/<variant>/<model>/eps_<eps>/
# ==============================================================================

PROTOCOL="mal-rep-field"

SCRIPTS=(
    "./run_crown_elemwise.sh"
    "./run_crown_batchrelu.sh"
    "./run_crown_batchsplit.sh"
    "./run_crown_naiveopt.sh"
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
echo "开始 mal-rep-field 协议 (3PC malicious, replicated) 全 MNIST 实证测试..."
echo "策略: 从小到大运行, 自动清理 Player-Data/*.sch 以节省磁盘."
echo "=============================================================================="

# Sanity check
if [ ! -f "./malicious-rep-field-party.x" ]; then
    echo "!! 警告: malicious-rep-field-party.x 未编译."
    echo "   请先执行: make -j4 malicious-rep-field-party.x"
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
echo "mal-rep-field 协议全部 MNIST 测试完成!"
echo "结果目录: crown-results/mal-rep-field/"
echo "=============================================================================="
