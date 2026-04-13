#!/bin/bash

# ==============================================================================
# 批量执行脚本: 全 MNIST 模型 spdz2k 协议实证测试
#
# spdz2k = 2PC dishonest-majority, MALICIOUS security
# 域:     环 Z_{2^k} (ring)
# 预处理: OT extension + sacrifice + MAC (在环上)
# 在线:   Beaver triples + MAC verification
# 关键对比: 与 mascot 相同的安全模型, 唯一差别是算术域 (素域 -> 环)
#          与 semi2k 相同的域, 差别是安全模型 (半诚实 -> 恶意)
#          完整 2x2 对比: semi/mascot/semi2k/spdz2k
#
# 注意: spdz2k 的 mnist_7layer_256 可能非常慢 (恶意+环, offline 最重)
# 结果保存在: crown-results/spdz2k_2pc/<variant>/<model>/eps_<eps>/
# ==============================================================================

PROTOCOL="spdz2k"

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
echo "开始 spdz2k 协议 (2PC malicious, ring) 全 MNIST 实证测试..."
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

        echo "清理 Player-Data/*.sch ..."
        rm -rf Player-Data/*.sch

        echo "准备进入下一个变体."
        sleep 3
    done
done

echo ""
echo "=============================================================================="
echo "spdz2k 协议全部 MNIST 测试完成!"
echo "结果目录: crown-results/spdz2k_2pc/"
echo "=============================================================================="
