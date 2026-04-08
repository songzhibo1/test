#!/bin/bash
# ============================================================
# EvalGuard CIFAR-10 — surrogate fine-tuning attack experiments
# rec_trigger vs own_trigger comparison
# ============================================================
#
# All knobs are environment variables — pass them on the command line
# to override the defaults below.  Examples:
#
#   # Paper-faithful parameters (论文 r_w=0.5%, delta=2.0)
#   TAG=paper RW=0.005 D_LOGIT=2.0 BETA=0.3 V_TEMP=5.0 \
#       ./run_cifar10_both.sh
#
#   # Amplified parameters for signal observation
#   TAG=amplified RW=0.1 D_LOGIT=5.0 BETA=0.7 V_TEMP=10.0 \
#       ./run_cifar10_both.sh
#
#   # CPU smoke test (small dataset, few epochs, runs in minutes)
#   TAG=smoke NQ=2000 DIST_EP=8 TEMPS="5" \
#       ./run_cifar10_both.sh
#
#   # GPU run when hardware is available
#   DEVICE=cuda TAG=gpu_amplified RW=0.1 D_LOGIT=5.0 \
#       ./run_cifar10_both.sh
#
# The TAG variable is appended to every checkpoint and result filename
# so that runs with different parameters never overwrite each other.
# ============================================================

set -e

# ----- Hardware ---------------------------------------------------------
DEVICE="${DEVICE:-cpu}"          # cpu | cuda

# ----- Watermark parameters --------------------------------------------
# 论文 (paper) defaults below; override via env vars when probing.
RW="${RW:-0.005}"                # watermark ratio   (paper: 0.005 = 0.5%)
D_LOGIT="${D_LOGIT:-2.0}"        # logit-space shift (paper: 2.0)
BETA="${BETA:-0.3}"              # safety factor     (paper: 0.3)
DELTA_MIN="${DELTA_MIN:-0.5}"    # reject triggers with delta < this
V_TEMP="${V_TEMP:-5.0}"          # verification softmax temperature

# ----- Distillation parameters -----------------------------------------
NQ="${NQ:-50000}"                # number of query inputs
DIST_EP="${DIST_EP:-80}"         # distillation epochs
DIST_LR="${DIST_LR:-0.002}"      # distillation lr
DIST_BATCH="${DIST_BATCH:-128}"  # distillation batch size
TEMPS="${TEMPS:-1,3,5,10}"       # distillation temperatures (CSV)

# ----- Surrogate fine-tuning parameters --------------------------------
FT_EP="${FT_EP:-20}"
FT_LR="${FT_LR:-0.0005}"
FT_FRACS="${FT_FRACS:-0.0,0.01,0.05,0.10}"

# ----- Trigger / verification setup ------------------------------------
LABEL_MODE="${LABEL_MODE:-soft}"             # soft | hard | both
TRIGGER_MODE="${TRIGGER_MODE:-both}"         # rec_trigger | own_trigger | both
TRIGGER_SIZES="${TRIGGER_SIZES:-0}"          # space-separated sweep; 0 = all

# ----- Output identification -------------------------------------------
TAG="${TAG:-paper}"              # appended to result/ckpt filenames

# ----- Models to sweep --------------------------------------------------
# Format: "<teacher_model>:<student_arch>"
MODEL_PAIRS="${MODEL_PAIRS:-cifar10_vgg11:vgg11 cifar10_resnet20:resnet20 cifar10_resnet20:vgg11}"

# ============================================================
# Echo configuration
# ============================================================
echo "============================================================"
echo " EvalGuard CIFAR-10 — surrogate_ft experiment"
echo " TAG          = ${TAG}"
echo " DEVICE       = ${DEVICE}"
echo " Watermark    : RW=${RW}, D_LOGIT=${D_LOGIT}, BETA=${BETA}, DELTA_MIN=${DELTA_MIN}, V_TEMP=${V_TEMP}"
echo " Distill      : NQ=${NQ}, DIST_EP=${DIST_EP}, DIST_LR=${DIST_LR}, BATCH=${DIST_BATCH}"
echo " Temperatures = ${TEMPS}"
echo " Label mode   = ${LABEL_MODE}"
echo " Trigger mode = ${TRIGGER_MODE}"
echo " FT           : EP=${FT_EP}, LR=${FT_LR}, FRACS=${FT_FRACS}"
echo " Trigger sizes= ${TRIGGER_SIZES}"
echo " Model pairs  = ${MODEL_PAIRS}"
echo "============================================================"

SCRIPT="python experiments.py"

for TS in ${TRIGGER_SIZES}; do
    echo ""
    echo "------------------------------------------------------------"
    echo " Trigger size = ${TS}  (0 = all available)"
    echo "------------------------------------------------------------"

    for PAIR in ${MODEL_PAIRS}; do
        TEACHER="${PAIR%%:*}"
        STUDENT="${PAIR##*:}"
        echo ""
        echo ">>> ${TEACHER} -> ${STUDENT}, trigger_size=${TS}, tag=${TAG}"

        ${SCRIPT} \
            --experiment surrogate_ft \
            --model "${TEACHER}" --student_arch "${STUDENT}" \
            --label_mode "${LABEL_MODE}" \
            --trigger_mode "${TRIGGER_MODE}" \
            --rec_trigger_size "${TS}" --own_trigger_size "${TS}" \
            --rw "${RW}" --delta_logit "${D_LOGIT}" --beta "${BETA}" \
            --delta_min "${DELTA_MIN}" \
            --verify_temperature "${V_TEMP}" \
            --temperatures "${TEMPS}" \
            --nq "${NQ}" \
            --dist_epochs "${DIST_EP}" --dist_lr "${DIST_LR}" --dist_batch "${DIST_BATCH}" \
            --ft_epochs "${FT_EP}" --ft_lr "${FT_LR}" --ft_fractions "${FT_FRACS}" \
            --device "${DEVICE}" \
            --tag "${TAG}"
    done
done

echo ""
echo "============================================================"
echo " CIFAR-10 sweep DONE   (tag=${TAG})"
echo " Results:     results/surrogate_ft/cifar10/"
echo " Checkpoints: checkpoints/distill/cifar10/"
echo "============================================================"