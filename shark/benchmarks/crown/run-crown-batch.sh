#!/bin/bash

# CROWN Batch Benchmark Runner
# 运行方式: ./run-crown-batch.sh [options]
#
# Options:
#   -c, --config <file>    Batch config file (default: batch_config_cifar.txt)
#   -m, --mode <mode>      malicious or semi-honest (default: both)
#   -t, --threads <n>      Number of threads (0 for default)
#   --model <name>         Model name
#   --eps <value>          Epsilon value

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 默认参数
CONFIG_FILE="$SCRIPT_DIR/batch_config_cifar.txt"
MODE="both"
THREADS=""
MODEL="eran_cifar_5layer_relu_100_best"
EPS="0.002"
NUM_LAYERS="5"
HIDDEN_DIM="100"
INPUT_DIM="3072"
OUTPUT_DIM="10"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --eps)
            EPS="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --input_dim)
            INPUT_DIM="$2"
            shift 2
            ;;
        --output_dim)
            OUTPUT_DIM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 检查配置文件
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

NUM_IMAGES=$(grep -v '^#' "$CONFIG_FILE" | grep -v '^$' | wc -l)

echo "========================================"
echo "CROWN Batch Benchmark"
echo "========================================"
echo "Config:      $CONFIG_FILE"
echo "Images:      $NUM_IMAGES"
echo "Model:       $MODEL"
echo "EPS:         $EPS"
echo "Mode:        $MODE"
echo "Threads:     ${THREADS:-default}"
echo "========================================"

# 设置线程数
if [[ -n "$THREADS" && "$THREADS" != "0" ]]; then
    export OMP_NUM_THREADS=$THREADS
fi

# 公共参数
COMMON_ARGS="--model=$MODEL --eps=$EPS --num_layers=$NUM_LAYERS --hidden_dim=$HIDDEN_DIM --input_dim=$INPUT_DIM --output_dim=$OUTPUT_DIM --batch_config=$CONFIG_FILE"

run_benchmark() {
    local mode=$1
    local sh_flag=""
    if [[ "$mode" == "semi-honest" ]]; then
        sh_flag="-sh"
    fi

    echo ""
    echo "Running $mode mode..."

    # Dealer
    "$SHARK_DIR/build/bin/crowntest_batch" -d $sh_flag $COMMON_ARGS &
    DEALER_PID=$!
    sleep 1

    # Server
    "$SHARK_DIR/build/bin/crowntest_batch" -1 $sh_flag $COMMON_ARGS &
    SERVER_PID=$!

    # Client (capture output)
    OUTPUT=$("$SHARK_DIR/build/bin/crowntest_batch" -0 $sh_flag $COMMON_ARGS 2>&1)

    wait $DEALER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null

    echo "$OUTPUT"

    # 提取时间
    TIME=$(echo "$OUTPUT" | grep "total_time:" | awk '{print $2}')
    WEIGHTS_TIME=$(echo "$OUTPUT" | grep "weights_input:" | awk '{print $2}')
    BATCH_TIME=$(echo "$OUTPUT" | grep "batch_computation:" | awk '{print $2}')

    echo ""
    echo "[$mode] Total: ${TIME}ms, Weights: ${WEIGHTS_TIME}ms, Computation: ${BATCH_TIME}ms"
}

# 运行基准测试
if [[ "$MODE" == "both" ]]; then
    echo ""
    echo "========== MALICIOUS MODE =========="
    run_benchmark "malicious"
    MAL_TIME=$(echo "$OUTPUT" | grep "total_time:" | awk '{print $2}')

    echo ""
    echo "========== SEMI-HONEST MODE =========="
    run_benchmark "semi-honest"
    SH_TIME=$(echo "$OUTPUT" | grep "total_time:" | awk '{print $2}')

    echo ""
    echo "========================================"
    echo "COMPARISON"
    echo "========================================"
    echo "Malicious:   ${MAL_TIME}ms"
    echo "Semi-honest: ${SH_TIME}ms"
    if [[ -n "$MAL_TIME" && -n "$SH_TIME" ]]; then
        IMPROVEMENT=$(echo "scale=1; ($MAL_TIME - $SH_TIME) * 100 / $MAL_TIME" | bc)
        echo "Improvement: ${IMPROVEMENT}%"
    fi
    echo "========================================"

elif [[ "$MODE" == "malicious" ]]; then
    run_benchmark "malicious"
elif [[ "$MODE" == "semi-honest" ]]; then
    run_benchmark "semi-honest"
fi

echo ""
echo "Batch benchmark completed!"
