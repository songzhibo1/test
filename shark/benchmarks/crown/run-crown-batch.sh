#!/bin/bash
set -e

# CROWN Batch Benchmark Runner
# 位置: shark/benchmarks/crown/run-crown-batch.sh
# 运行: cd shark && ./benchmarks/crown/run-crown-batch.sh [options]

# 获取脚本所在目录和 shark 根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 切换到 shark 目录
cd "$SHARK_DIR"

# ==================== 默认参数 ====================
CONFIG_FILE="$SCRIPT_DIR/batch_config_cifar.txt"
MODE="both"
THREADS="4"
MODEL="eran_cifar_5layer_relu_100_best"
EPS="0.002"
NUM_LAYERS="5"
HIDDEN_DIM="100"
INPUT_DIM="3072"
OUTPUT_DIM="10"

# ==================== 解析参数 ====================
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
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -c, --config <file>    Batch config file (default: batch_config_cifar.txt)"
            echo "  -m, --mode <mode>      malicious, semi-honest, or both (default: both)"
            echo "  -t, --threads <n>      Number of threads (default: 4)"
            echo "  --model <name>         Model name"
            echo "  --eps <value>          Epsilon value (default: 0.002)"
            echo "  --num_layers <n>       Number of layers"
            echo "  --hidden_dim <n>       Hidden dimension"
            echo "  --input_dim <n>        Input dimension"
            echo "  --output_dim <n>       Output dimension"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==================== 检查配置文件 ====================
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

NUM_IMAGES=$(grep -v '^#' "$CONFIG_FILE" | grep -v '^$' | wc -l)

# ==================== 检查可执行文件 ====================
if [[ ! -f "./build/crowntest_batch" ]]; then
    echo "Error: ./build/crowntest_batch not found"
    echo "Please build with: cd build && cmake .. && make crowntest_batch"
    exit 1
fi

echo "========================================"
echo "CROWN Batch Benchmark"
echo "========================================"
echo "Working dir: $SHARK_DIR"
echo "Config:      $CONFIG_FILE"
echo "Images:      $NUM_IMAGES"
echo "Model:       $MODEL"
echo "EPS:         $EPS"
echo "Mode:        $MODE"
echo "Threads:     $THREADS"
echo "========================================"

# ==================== 设置线程数 ====================
if [[ "$THREADS" != "0" ]]; then
    export OMP_NUM_THREADS=$THREADS
fi

# ==================== 公共参数 ====================
COMMON_ARGS="--model=$MODEL --eps=$EPS --num_layers=$NUM_LAYERS --hidden_dim=$HIDDEN_DIM --input_dim=$INPUT_DIM --output_dim=$OUTPUT_DIM --batch_config=$CONFIG_FILE"

# ==================== 清理函数 ====================
cleanup() {
    pkill -f "crowntest_batch" 2>/dev/null || true
}
trap cleanup EXIT

# ==================== 运行函数 ====================
# Party IDs: 0=SERVER, 1=CLIENT, 2=DEALER
# 格式: ./crowntest_batch <party> [ip] [--semi-honest] [--other-options]

IP_ADDRESS="127.0.0.1"

run_benchmark() {
    local mode=$1
    local sh_flag=""
    if [[ "$mode" == "semi-honest" ]]; then
        sh_flag="--semi-honest"
    fi

    echo ""
    echo "Running $mode mode..."
    echo "Command: ./build/crowntest_batch <party> [ip] $sh_flag $COMMON_ARGS"

    # Dealer (party 2) - 无需 IP
    ./build/crowntest_batch 2 $sh_flag $COMMON_ARGS &
    DEALER_PID=$!
    sleep 2

    # Server (party 0) - 需要 IP
    ./build/crowntest_batch 0 $IP_ADDRESS $sh_flag $COMMON_ARGS &
    SERVER_PID=$!
    sleep 1

    # Client (party 1) - 需要 IP，捕获输出
    OUTPUT=$(./build/crowntest_batch 1 $IP_ADDRESS $sh_flag $COMMON_ARGS 2>&1)

    wait $DEALER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true

    echo "$OUTPUT"

    # 提取时间
    TIME=$(echo "$OUTPUT" | grep "total_time:" | awk '{print $2}')
    WEIGHTS_TIME=$(echo "$OUTPUT" | grep "weights_input:" | awk '{print $2}')
    BATCH_TIME=$(echo "$OUTPUT" | grep "batch_computation:" | awk '{print $2}')

    echo ""
    echo "[$mode] Total: ${TIME}ms, Weights: ${WEIGHTS_TIME}ms, Computation: ${BATCH_TIME}ms"
}

# ==================== 运行基准测试 ====================
if [[ "$MODE" == "both" ]]; then
    echo ""
    echo "========== MALICIOUS MODE =========="
    run_benchmark "malicious"
    MAL_OUTPUT="$OUTPUT"
    MAL_TIME=$(echo "$MAL_OUTPUT" | grep "total_time:" | awk '{print $2}')

    echo ""
    echo "========== SEMI-HONEST MODE =========="
    run_benchmark "semi-honest"
    SH_OUTPUT="$OUTPUT"
    SH_TIME=$(echo "$SH_OUTPUT" | grep "total_time:" | awk '{print $2}')

    echo ""
    echo "========================================"
    echo "COMPARISON"
    echo "========================================"
    echo "Malicious:   ${MAL_TIME}ms"
    echo "Semi-honest: ${SH_TIME}ms"
    if [[ -n "$MAL_TIME" && -n "$SH_TIME" && "$MAL_TIME" != "0" ]]; then
        IMPROVEMENT=$(echo "scale=1; ($MAL_TIME - $SH_TIME) * 100 / $MAL_TIME" | bc 2>/dev/null || echo "N/A")
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
