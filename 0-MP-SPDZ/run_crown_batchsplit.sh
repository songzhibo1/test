#!/bin/bash
set -e

# ============================================================
# CROWN Verification Runner (Batch Sign-Split) for MP-SPDZ
#
# Runs crown_batchsplit.mpc -- sign-split optimization that halves
# comparison count vs batchrelu while keeping matrix-level batch ops.
#
# Results saved to: crown-results/<protocol>/crown_batchsplit/<model>/eps_<eps>/
#
# Usage:
#   ./run_crown_batchsplit.sh [protocol] [model_config]
#
# Examples:
#   ./run_crown_batchsplit.sh semi
#   ./run_crown_batchsplit.sh semi mnist_3layer_20
#   ./run_crown_batchsplit.sh mascot mnist_5layer_256
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# MPC variant identifier
MPC_VARIANT="crown_batchsplit"
MPC_SOURCE="crown/crown_batchsplit"

# Default protocol
PROTOCOL="${1:-semi}"

# Model configuration presets
MODEL_PRESET="${2:-mnist_3layer_20}"

# ==================== Crown MPC data base path ====================
CROWN_DATA_BASE="${CROWN_DATA_BASE:-/usr/src/crown/shark_sh/shark_crown_ml/crown_mpc_data}"

# ==================== Model Configurations ====================
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

# Paths
WEIGHTS_FILE="${CROWN_DATA_BASE}/${DATA_FOLDER}/weights/weights.dat"
INPUT_FILE="${CROWN_DATA_BASE}/${DATA_FOLDER}/images/${IMAGE_ID}.bin"

# ==================== Results directory (variant-specific) ====================
RESULTS_BASE_DIR="crown-results/${PROTOCOL}/${MPC_VARIANT}/${MODEL_PRESET}/eps_${EPS}"
mkdir -p "$RESULTS_BASE_DIR"

RESULT_LOG="${RESULTS_BASE_DIR}/image_${IMAGE_ID}_${PROTOCOL}_log.txt"
RESULT_SUMMARY="${RESULTS_BASE_DIR}/image_${IMAGE_ID}_${PROTOCOL}_summary.txt"

echo "========================================"
echo "CROWN Verification on MP-SPDZ (Batch Sign-Split)"
echo "========================================"
echo "MPC variant:  $MPC_VARIANT"
echo "Protocol:     $PROTOCOL"
echo "Model:        $MODEL_PRESET ($DATA_FOLDER)"
echo "Layers:       $NUM_LAYERS"
echo "Layer dims:   $LAYER_DIMS"
echo "Eps:          $EPS"
echo "True label:   $TRUE_LABEL"
echo "Target label: $TARGET_LABEL"
echo "Image ID:     $IMAGE_ID"
echo "Weights:      $WEIGHTS_FILE"
echo "Input:        $INPUT_FILE"
echo "Results dir:  $RESULTS_BASE_DIR"
echo "========================================"

# ==================== Step 1: Prepare Data ====================
echo ""
echo "[Step 1] Preparing input data..."

if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "ERROR: Weights file not found: $WEIGHTS_FILE"
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
echo "[Step 2] Compiling ${MPC_SOURCE}.mpc..."

COMPILE_ARGS="${MPC_SOURCE} $NUM_LAYERS $LAYER_DIMS"
echo "Compile args: $COMPILE_ARGS"

python3 ./compile.py $COMPILE_ARGS

# Program name: crown/crown_batchsplit-<num_layers>-<d0>-...-<dN>
PROGRAM_NAME="${MPC_SOURCE}-${NUM_LAYERS}"
for d in $LAYER_DIMS; do
    PROGRAM_NAME="${PROGRAM_NAME}-${d}"
done

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
        echo "Available: emulate, semi, mascot, rep-field, shamir, mal-rep-field, semi2k, spdz2k, rep-ring, hemi, soho"
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

# Run and capture output
bash "$SCRIPT" "$PROGRAM_NAME" 2>&1 | tee "$RESULT_LOG"

echo "========================================"
echo "Done!"

# ==================== Step 4: Save Summary ====================
echo ""
echo "[Step 4] Saving results..."

{
    echo "============================================"
    echo "CROWN Verification Summary (Batch Sign-Split)"
    echo "============================================"
    echo "Date:         $(date '+%Y-%m-%d %H:%M:%S')"
    echo "MPC variant:  $MPC_VARIANT"
    echo "Model:        $MODEL_PRESET ($DATA_FOLDER)"
    echo "Layers:       $NUM_LAYERS"
    echo "Layer dims:   $LAYER_DIMS"
    echo "Hidden dim:   $HIDDEN_DIM"
    echo "Input dim:    $INPUT_DIM"
    echo "Output dim:   $OUTPUT_DIM"
    echo "EPS:          $EPS"
    echo "True label:   $TRUE_LABEL"
    echo "Target label: $TARGET_LABEL"
    echo "Image ID:     $IMAGE_ID"
    echo "Protocol:     $PROTOCOL"
    echo "--------------------------------------------"
    echo "Computation Results:"
    grep -E "MPC LB:|MPC UB:|Robust:" "$RESULT_LOG" 2>/dev/null || echo "  (no results found)"
    echo "--------------------------------------------"
    echo "Performance:"
    grep -E "^Time =|^Data sent =|^Global data sent =" "$RESULT_LOG" 2>/dev/null || echo "  (no performance data)"
    echo "============================================"
} > "$RESULT_SUMMARY"

echo "Results saved:"
echo "  Full log:  $RESULT_LOG"
echo "  Summary:   $RESULT_SUMMARY"
echo ""
cat "$RESULT_SUMMARY"
