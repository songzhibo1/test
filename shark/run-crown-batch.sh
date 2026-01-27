#!/bin/bash
set -e

# ==================== CROWN Batch Benchmark Runner ====================
# 位置: shark/run-crown-batch.sh (与 run-crown-benchmarks-local-new.sh 同级)
# 功能: 一次 MPC 会话中批量处理多个图片
# 运行: ./run-crown-batch.sh [options]

# ==================== 运行配置 ====================
THREADS_LIST=(4)
RUN_MODE="both"  # "both", "malicious", "semi-honest"

# ==================== 网络配置 ====================
IP_ADDRESS="127.0.0.1"

# ==================== 模型配置 ====================
# 格式: "num_layers hidden_dim input_dim output_dim"
declare -A MODEL_CONFIGS
MODEL_CONFIGS["vnncomp_mnist_3layer_relu_256_best"]="3 256 784 10"
MODEL_CONFIGS["vnncomp_mnist_5layer_relu_256_best"]="5 256 784 10"
MODEL_CONFIGS["vnncomp_mnist_7layer_relu_256_best"]="7 256 784 10"
MODEL_CONFIGS["cifar_6layer_relu_2048_best"]="6 2048 3072 10"
MODEL_CONFIGS["eran_cifar_5layer_relu_100_best"]="5 100 3072 10"
MODEL_CONFIGS["eran_cifar_7layer_relu_100_best"]="7 100 3072 10"
MODEL_CONFIGS["eran_cifar_10layer_relu_200_best"]="10 200 3072 10"

# ==================== 测试模型列表 ====================
MODELS_TO_TEST=(
    "eran_cifar_5layer_relu_100_best"
)

# EPS 列表
EPS_LIST=(0.002)

# ==================== 结果目录 ====================
RESULTS_BASE_DIR="crown-batch-results"
mkdir -p "$RESULTS_BASE_DIR"

# ==================== 辅助函数 ====================
bytes_to_human() {
    local bytes=$1
    if [ $bytes -ge 1073741824 ]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1073741824}") GB"
    elif [ $bytes -ge 1048576 ]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1048576}") MB"
    elif [ $bytes -ge 1024 ]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1024}") KB"
    else
        echo "$bytes bytes"
    fi
}

cleanup_processes() {
    echo "Cleaning up any remaining benchmark processes..."
    pkill -f "crowntest_batch" 2>/dev/null || true
    sleep 2
}

trap cleanup_processes EXIT

# 确定要运行的模式列表
MODES_TO_RUN=()
if [ "$RUN_MODE" = "both" ]; then
    MODES_TO_RUN=("malicious" "semi-honest")
elif [ "$RUN_MODE" = "malicious" ]; then
    MODES_TO_RUN=("malicious")
elif [ "$RUN_MODE" = "semi-honest" ]; then
    MODES_TO_RUN=("semi-honest")
fi

echo "========================================"
echo "CROWN Batch Benchmark Runner"
echo "========================================"
echo "Results Dir:  $RESULTS_BASE_DIR"
echo "Threads List: ${THREADS_LIST[*]}"
echo "Run Mode:     $RUN_MODE"
echo "Modes:        ${MODES_TO_RUN[*]}"
echo "========================================"
echo ""

# ==================== 主循环 ====================
for model in "${MODELS_TO_TEST[@]}"; do
    config=(${MODEL_CONFIGS[$model]})
    NUM_LAYERS=${config[0]}
    HIDDEN_DIM=${config[1]}
    INPUT_DIM=${config[2]}
    OUTPUT_DIM=${config[3]}

    # 批量配置文件路径
    BATCH_CONFIG="shark_crown_ml/crown_mpc_data/${model}/batch_config.txt"

    if [ ! -f "$BATCH_CONFIG" ]; then
        echo "WARNING: Batch config not found: $BATCH_CONFIG"
        echo "Skipping model: $model"
        continue
    fi

    NUM_IMAGES=$(grep -v '^#' "$BATCH_CONFIG" | grep -v '^$' | wc -l)

    echo ""
    echo "########################################################"
    echo "# Model: $model"
    echo "# Layers: $NUM_LAYERS, Hidden: $HIDDEN_DIM"
    echo "# Input: $INPUT_DIM, Output: $OUTPUT_DIM"
    echo "# Batch config: $BATCH_CONFIG"
    echo "# Images: $NUM_IMAGES"
    echo "########################################################"
    echo ""

    mkdir -p "$RESULTS_BASE_DIR/$model"

    for eps in "${EPS_LIST[@]}"; do
        echo "=========================================="
        echo "Testing EPS = $eps"
        echo "=========================================="

        mkdir -p "$RESULTS_BASE_DIR/$model/eps_$eps"

        for current_mode in "${MODES_TO_RUN[@]}"; do
            if [ "$current_mode" = "semi-honest" ]; then
                SH_FLAG="--semi-honest"
                MODE_SUFFIX="_sh"
                MODE_DISPLAY="Semi-Honest"
            else
                SH_FLAG=""
                MODE_SUFFIX="_m"
                MODE_DISPLAY="Malicious"
            fi

            for THREADS in "${THREADS_LIST[@]}"; do
                THREAD_SUFFIX="_t${THREADS}"

                echo ""
                echo ">>> Running $MODE_DISPLAY mode, Threads=$THREADS"

                # 检查是否已存在结果
                RESULT_FILE="$RESULTS_BASE_DIR/$model/eps_$eps/batch${MODE_SUFFIX}${THREAD_SUFFIX}_client.txt"
                if [ -f "$RESULT_FILE" ]; then
                    echo "[SKIP] Results already exist: $RESULT_FILE"
                    continue
                fi

                # 构建参数
                CUSTOM_ARGS="--model=$model --num_layers=$NUM_LAYERS --hidden_dim=$HIDDEN_DIM --input_dim=$INPUT_DIM --output_dim=$OUTPUT_DIM --eps=$eps --batch_config=$BATCH_CONFIG $SH_FLAG"

                cleanup_processes

                echo "[$(date +'%H:%M:%S')] Starting DEALER..."

                # Dealer (party 2)
                if [ "$THREADS" = "0" ]; then
                    ./build/crowntest_batch 2 $CUSTOM_ARGS &> "$RESULTS_BASE_DIR/$model/eps_$eps/batch${MODE_SUFFIX}${THREAD_SUFFIX}_dealer.txt" &
                else
                    OMP_NUM_THREADS=$THREADS ./build/crowntest_batch 2 $CUSTOM_ARGS &> "$RESULTS_BASE_DIR/$model/eps_$eps/batch${MODE_SUFFIX}${THREAD_SUFFIX}_dealer.txt" &
                fi
                DEALER_PID=$!
                sleep 3

                echo "[$(date +'%H:%M:%S')] Starting SERVER and CLIENT..."

                # Server (party 0)
                if [ "$THREADS" = "0" ]; then
                    ./build/crowntest_batch 0 $IP_ADDRESS $CUSTOM_ARGS &> "$RESULTS_BASE_DIR/$model/eps_$eps/batch${MODE_SUFFIX}${THREAD_SUFFIX}_server.txt" &
                else
                    OMP_NUM_THREADS=$THREADS ./build/crowntest_batch 0 $IP_ADDRESS $CUSTOM_ARGS &> "$RESULTS_BASE_DIR/$model/eps_$eps/batch${MODE_SUFFIX}${THREAD_SUFFIX}_server.txt" &
                fi
                SERVER_PID=$!
                sleep 2

                # Client (party 1)
                if [ "$THREADS" = "0" ]; then
                    ./build/crowntest_batch 1 $IP_ADDRESS $CUSTOM_ARGS &> "$RESULT_FILE"
                else
                    OMP_NUM_THREADS=$THREADS ./build/crowntest_batch 1 $IP_ADDRESS $CUSTOM_ARGS &> "$RESULT_FILE"
                fi
                CLIENT_EXIT_CODE=$?

                wait $DEALER_PID 2>/dev/null || true
                wait $SERVER_PID 2>/dev/null || true

                echo "[$(date +'%H:%M:%S')] Batch computation completed!"

                if [ $CLIENT_EXIT_CODE -ne 0 ]; then
                    echo "WARNING: Client exited with code $CLIENT_EXIT_CODE"
                fi

                # 显示结果摘要
                echo ""
                echo "--- Results Summary ---"
                grep -E "(Total images|Verified|total_time|weights_input|batch_computation)" "$RESULT_FILE" 2>/dev/null || echo "(No summary found)"
                echo ""

                sleep 3
            done
        done
    done

    # 创建模型摘要
    {
        echo "Batch Model Summary: $model"
        echo "=========================="
        echo "Layers: $NUM_LAYERS, Hidden: $HIDDEN_DIM"
        echo "Input: $INPUT_DIM, Output: $OUTPUT_DIM"
        echo "Batch config: $BATCH_CONFIG"
        echo "Images: $NUM_IMAGES"
        echo ""
        echo "--- Results ---"
        for eps in "${EPS_LIST[@]}"; do
            echo "EPS = $eps:"
            for mode in "${MODES_TO_RUN[@]}"; do
                if [ "$mode" = "semi-honest" ]; then
                    suffix="_sh"
                else
                    suffix="_m"
                fi
                for t in "${THREADS_LIST[@]}"; do
                    f="$RESULTS_BASE_DIR/$model/eps_$eps/batch${suffix}_t${t}_client.txt"
                    if [ -f "$f" ]; then
                        echo "  [$mode, threads=$t]:"
                        grep -E "(Total images|Verified|total_time|batch_computation)" "$f" 2>/dev/null | sed 's/^/    /'
                    fi
                done
            done
        done
    } > "$RESULTS_BASE_DIR/$model/batch_summary.txt"
done

echo ""
echo "=========================================================="
echo "All batch tests completed!"
echo "Results saved in: $RESULTS_BASE_DIR/"
echo "=========================================================="
