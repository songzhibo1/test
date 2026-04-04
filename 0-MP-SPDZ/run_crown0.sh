#!/bin/bash
set -e

# ============================================================
# CROWN Verification Runner for MP-SPDZ
#
# This script:
#   1. Prepares input data from Crown MPC binary files
#   2. Compiles the crown.mpc program
#   3. Runs it with the selected protocol
#
# Usage:
#   ./run_crown.sh [protocol] [model_config]
#
# Examples:
#   ./run_crown.sh semi                           # Default: 3-layer MNIST, semi-honest
#   ./run_crown.sh mascot                         # MASCOT protocol
#   ./run_crown.sh rep-field                      # Replicated secret sharing
#   ./run_crown.sh semi mnist_3layer_20           # 3-layer MNIST hidden=20
#   ./run_crown.sh semi cifar_5layer_100          # 5-layer CIFAR hidden=100
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default protocol
PROTOCOL="${1:-semi}"

# Model configuration presets
MODEL_PRESET="${2:-mnist_3layer_20}"

# ==================== Crown MPC data base path ====================
# Docker 环境路径:  /usr/src/MP-SPDZ  运行SPDZ
#                   /usr/src/crown/shark_sh/shark_crown_ml/crown_mpc_data  数据目录
# 可通过环境变量 CROWN_DATA_BASE 覆盖
CROWN_DATA_BASE="${CROWN_DATA_BASE:-/usr/src/crown/shark_sh/shark_crown_ml/crown_mpc_data}"

# ==================== Model Configurations ====================
# Format: num_layers hidden_dim input_dim output_dim data_folder_name
declare -A CONFIGS

# MNIST small models (hidden=20, input_dim=784)
CONFIGS["mnist_2layer_20"]="2 20 784 10 mnist_2layer_relu_20_best"
CONFIGS["mnist_3layer_20"]="3 20 784 10 mnist_3layer_relu_20_best"

# MNIST test models (hidden=20, input_dim=784)
CONFIGS["test_mnist_4layer_20"]="4 20 784 10 test_mnist_4layer_relu_20_best"
CONFIGS["test_mnist_5layer_20"]="5 20 784 10 test_mnist_5layer_relu_20_best"
CONFIGS["test_mnist_6layer_20"]="6 20 784 10 test_mnist_6layer_relu_20_best"
CONFIGS["test_mnist_8layer_20"]="8 20 784 10 test_mnist_8layer_relu_20_best"

# MNIST large models (hidden=256, input_dim=784)
CONFIGS["mnist_3layer_256"]="3 256 784 10 vnncomp_mnist_3layer_relu_256_best"
CONFIGS["mnist_5layer_256"]="5 256 784 10 vnncomp_mnist_5layer_relu_256_best"
CONFIGS["mnist_7layer_256"]="7 256 784 10 vnncomp_mnist_7layer_relu_256_best"

# CIFAR models (input_dim=3072)
CONFIGS["cifar_6layer_2048"]="6 2048 3072 10 cifar_6layer_relu_2048_best"
CONFIGS["cifar_5layer_100"]="5 100 3072 10 eran_cifar_5layer_relu_100_best"
CONFIGS["cifar_7layer_100"]="7 100 3072 10 eran_cifar_7layer_relu_100_best"
CONFIGS["cifar_10layer_200"]="10 200 3072 10 eran_cifar_10layer_relu_200_best"

# ==================== Parse Configuration ====================
if [ -z "${CONFIGS[$MODEL_PRESET]}" ]; then
    echo "ERROR: Unknown model preset '$MODEL_PRESET'"
    echo "Available presets:"
    for key in "${!CONFIGS[@]}"; do
        echo "  $key -> ${CONFIGS[$key]}"
    done
    exit 1
fi

config_parts=(${CONFIGS[$MODEL_PRESET]})
NUM_LAYERS=${config_parts[0]}
HIDDEN_DIM=${config_parts[1]}
INPUT_DIM=${config_parts[2]}
OUTPUT_DIM=${config_parts[3]}
DATA_FOLDER=${config_parts[4]}

# Build layer_dims: [input_dim, hidden, hidden, ..., output_dim]
LAYER_DIMS="$INPUT_DIM"
for ((i=0; i<NUM_LAYERS-1; i++)); do
    LAYER_DIMS="$LAYER_DIMS $HIDDEN_DIM"
done
LAYER_DIMS="$LAYER_DIMS $OUTPUT_DIM"

# Test parameters (configurable via env vars)
EPS="${CROWN_EPS:-0.03}"
TRUE_LABEL="${CROWN_TRUE_LABEL:-7}"
TARGET_LABEL="${CROWN_TARGET_LABEL:-6}"
IMAGE_ID="${CROWN_IMAGE_ID:-0}"

# Scale eps for compile-time integer argument (eps * 100000)
EPS_SCALED=$(python3 -c "print(int(${EPS} * 100000))")

# Paths
WEIGHTS_FILE="${CROWN_DATA_BASE}/${DATA_FOLDER}/weights/weights.dat"
INPUT_FILE="${CROWN_DATA_BASE}/${DATA_FOLDER}/images/${IMAGE_ID}.bin"

echo "========================================"
echo "CROWN Verification on MP-SPDZ"
echo "========================================"
echo "Protocol:     $PROTOCOL"
echo "Model:        $MODEL_PRESET ($DATA_FOLDER)"
echo "Layers:       $NUM_LAYERS"
echo "Layer dims:   $LAYER_DIMS"
echo "Eps:          $EPS (scaled: $EPS_SCALED)"
echo "True label:   $TRUE_LABEL"
echo "Target label: $TARGET_LABEL"
echo "Image ID:     $IMAGE_ID"
echo "Weights:      $WEIGHTS_FILE"
echo "Input:        $INPUT_FILE"
echo "========================================"

# ==================== Step 1: Prepare Data ====================
echo ""
echo "[Step 1] Preparing input data..."

if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "ERROR: Weights file not found: $WEIGHTS_FILE"
    echo "Please run the data conversion script first:"
    echo "  cd /usr/src/crown/shark_sh/shark_crown_ml && python Convert-for-crown-mpc.py"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

python3 Programs/Source/crown/crown_prepare_data.py \
    --weights-file "$WEIGHTS_FILE" \
    --input-file "$INPUT_FILE" \
    --layer-dims $LAYER_DIMS \
    --eps "$EPS" \
    --true-label "$TRUE_LABEL" \
    --target-label "$TARGET_LABEL" \
    --output-dir Player-Data

echo "Data preparation complete."

# ==================== Step 2: Compile ====================
echo ""
echo "[Step 2] Compiling crown0.mpc..."

COMPILE_ARGS="crown/crown0 $NUM_LAYERS $LAYER_DIMS $EPS_SCALED $TRUE_LABEL $TARGET_LABEL"
echo "Compile args: $COMPILE_ARGS"

python3 ./compile.py  $COMPILE_ARGS

# The compiled program name includes the args
PROGRAM_NAME="crown/crown-${NUM_LAYERS}"
for d in $LAYER_DIMS; do
    PROGRAM_NAME="${PROGRAM_NAME}-${d}"
done
PROGRAM_NAME="${PROGRAM_NAME}-${EPS_SCALED}-${TRUE_LABEL}-${TARGET_LABEL}"

echo "Compiled program: $PROGRAM_NAME"

# ==================== Step 3: Run ====================
echo ""
echo "[Step 3] Running with protocol: $PROTOCOL"

# Map protocol names to scripts
case "$PROTOCOL" in
    emulate|emu)
        SCRIPT="Scripts/emulate.sh"
        ;;
    semi|semi-honest)
        SCRIPT="Scripts/semi.sh"
        ;;
    mascot)
        SCRIPT="Scripts/mascot.sh"
        ;;
    rep-field|replicated)
        SCRIPT="Scripts/rep-field.sh"
        ;;
    shamir)
        SCRIPT="Scripts/shamir.sh"
        ;;
    mal-rep-field|malicious-replicated)
        SCRIPT="Scripts/mal-rep-field.sh"
        ;;
    semi2k)
        SCRIPT="Scripts/semi2k.sh"
        ;;
    spdz2k)
        SCRIPT="Scripts/spdz2k.sh"
        ;;
    rep-ring)
        SCRIPT="Scripts/rep-ring.sh"
        ;;
    ps-rep-field)
        SCRIPT="Scripts/ps-rep-field.sh"
        ;;
    sy-rep-field)
        SCRIPT="Scripts/sy-rep-field.sh"
        ;;
    hemi)
        SCRIPT="Scripts/hemi.sh"
        ;;
    soho)
        SCRIPT="Scripts/soho.sh"
        ;;
    *)
        echo "ERROR: Unknown protocol '$PROTOCOL'"
        echo "Available: semi, mascot, rep-field, shamir, mal-rep-field, semi2k, spdz2k, rep-ring, hemi, soho"
        exit 1
        ;;
esac

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Protocol script not found: $SCRIPT"
    echo "Make sure you have compiled the MP-SPDZ binaries."
    exit 1
fi

echo "Running: $SCRIPT $PROGRAM_NAME"
echo "========================================"
bash "$SCRIPT" "$PROGRAM_NAME"
echo "========================================"
echo "Done!"
