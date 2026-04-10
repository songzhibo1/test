#!/bin/bash

# ==============================================================================
# 自动化批量执行脚本：全模型 semi 协议实证测试 (带自动磁盘清理)
# ==============================================================================

# 5种 MPC 变体执行脚本
SCRIPTS=(
    "./run_crown_elemwise.sh"
    "./run_crown_batchrelu.sh"
    "./run_crown_batchsplit.sh"
    "./run_crown_navieopt.sh"
    "./run_crown_naive.sh"
)

# 测试用例排序：从小到大
TEST_CASES=(
    "mnist_2layer_20 0.04500 0 7 4"
    "mnist_3layer_20 0.03000 0 7 6"
    "mnist_3layer_256 0.01500 0 7 3"
    "mnist_5layer_256 0.01500 0 7 8"
    "mnist_7layer_256 0.01500 0 7 3"
    "cifar_5layer_100 0.00200 0 3 6"
    "cifar_7layer_100 0.00100 5 6 0"
    "cifar_10layer_200 0.00100 1 8 4"
)

echo "🚀 开始全模型 semi 协议实证测试..."
echo "📊 策略：从小到大运行，并开启自动磁盘空间回收。"
echo "=============================================================================="

for case in "${TEST_CASES[@]}"; do
    read -r MODEL EPS ID TRUE TARGET <<< "$case"

    echo ""
    echo "##############################################################################"
    echo "▶ [CASE START] 模型: $MODEL | EPS: $EPS"
    echo "##############################################################################"

    for SCRIPT in "${SCRIPTS[@]}"; do
        echo "--------------------------------------------------------"
        echo ">>> 执行变体: $SCRIPT (Protocol: semi)"
        echo "--------------------------------------------------------"
        
        chmod +x "$SCRIPT"

        # 1. 运行 semi 协议任务
        CROWN_EPS=$EPS \
        CROWN_IMAGE_ID=$ID \
        CROWN_TRUE_LABEL=$TRUE \
        CROWN_TARGET_LABEL=$TARGET \
        $SCRIPT semi "$MODEL"
        
        # 2. 【核心改进】立即清理预处理产生的巨大文件
        # semi 协议产生的 *.sch 文件通常存放在 Player-Data 目录下
        echo "🧹 任务结束，正在清理 Player-Data 下的预处理缓存..."
        rm -rf Player-Data/*.sch
        
        # 也可以清理二进制编译产物以进一步节省空间（可选）
        # rm -f Programs/Bytecode/*.bc
        
        echo "✅ 空间已释放，准备进入下一个变体。"
        sleep 3
    done
done

echo ""
echo "=============================================================================="
echo "🎉 所有规模案例的 semi 测试（含自动清理）已全部完成！"
echo "请查看 crown-results/semi 下的 summary 文件进行分析。"
echo "=============================================================================="