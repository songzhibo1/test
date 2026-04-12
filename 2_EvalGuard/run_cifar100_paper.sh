#!/bin/bash
# =============================================================================
# EvalGuard — CIFAR-100 Paper Experiments
# =============================================================================
#
# Key difference from CIFAR-10:
#   - DIST_EP=120 (vs 80): 100 classes require more epochs to converge.
#     At 80 epochs, vgg11→vgg11 only reaches ~49% (teacher 70.8%).
#   - delta_logit sweep goes up to 10.0: 100 classes dilute softmax signal
#     by ~10x, stronger delta may be needed.
#
# Usage:
#   GPU_IDS="0,1,2,3,4,5" ./run_cifar100_paper.sh
#   GPU_IDS="0,1,2,3,4,5" STEPS="exp1" ./run_cifar100_paper.sh
#   GPU_IDS="0,1,2,3,4,5" DRY_RUN=1 ./run_cifar100_paper.sh
#
# Available STEPS:
#   exp1  Main table (Table VIII): 3 pairs × 4 temps               12 jobs
#   exp2  rw sweep:          {0.05, 0.10, 0.20}                     3 jobs
#   exp3  delta_logit sweep: {3.0, 5.0, 7.0, 10.0}                  4 jobs
#   exp4  False positive:    hard-label baseline                     2 jobs
#                                                           Total = 21 jobs
# =============================================================================

cd "$(dirname "$0")"

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
GPU_IDS="${GPU_IDS:-0}"
STEPS="${STEPS:-exp1 exp2 exp3 exp4}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs_cifar100}"
mkdir -p "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_LIST[@]}

# ─────────────────────────────────────────────────────────────────
# CIFAR-100 hyper-parameters
# ─────────────────────────────────────────────────────────────────
NQ=50000
DIST_EP=160           # 120 epochs for CIFAR-100 (100 classes, harder to converge)
DIST_LR=0.002
DIST_BATCH=128
FT_EP=40
FT_LR=0.0005
FT_FRACS="0.0,0.01,0.05,0.10"

RW=0.1
D_LOGIT=5.0
BETA=0.5
V_TEMP=10.0
DELTA_MIN=0.5

SWEEP_T=5             # baseline temperature for parameter sweeps

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
step_enabled() {
    for x in ${STEPS}; do [ "$x" = "$1" ] && return 0; done
    return 1
}

JOB_FILE=$(mktemp /tmp/evalguard_c100_jobs.XXXXXX)
JOB_COUNT=0

add_job() {
    local tag="$1"; shift
    printf "%d\t%s\t%s\n" "${JOB_COUNT}" "${tag}" "$*" >> "${JOB_FILE}"
    JOB_COUNT=$((JOB_COUNT + 1))
}

# common_args MODEL STUDENT TEMPS RW D BETA TAG [EXTRA...]
common_args() {
    local model=$1 student=$2 temps=$3 rw=$4 d=$5 beta=$6 tag=$7
    shift 7; local extra="$*"
    echo "--experiment surrogate_ft \
--model ${model} --student_arch ${student} \
--label_mode soft --trigger_mode rec_trigger --own_data_source trainset \
--rw ${rw} --delta_logit ${d} --beta ${beta} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${temps} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${tag} ${extra}"
}

# ─────────────────────────────────────────────────────────────────
# Pre-download guard
# ─────────────────────────────────────────────────────────────────
if [ "${DRY_RUN}" != "1" ]; then
    echo "[prep] warming CIFAR-100 cache..."
    python - <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from evalguard.configs import cifar100_data, cifar100_resnet20, cifar100_vgg11, cifar100_resnet56, create_student
_ = cifar100_data();  print("  CIFAR-100 data ready.")
for fn in [cifar100_resnet20, cifar100_vgg11, cifar100_resnet56]:
    _ = fn(pretrained=True)
for arch in ["resnet20", "vgg11"]:
    _ = create_student(num_classes=100, arch=arch)
print("  Models cached.")
PY
fi


# =============================================================================
# EXP 1 — Table VIII: CIFAR-100 main results
# =============================================================================
#   3 architecture pairs × 4 temperatures → 12 jobs
#   Pairs: resnet56→resnet56, resnet56→resnet20, vgg11→vgg11
#   T = {1, 3, 5, 10}  (same range as CIFAR-10 for direct comparison)
#
#   Note: T=1 is expected to fail on CIFAR-100 (100-class softmax saturation).
#   Including it demonstrates the temperature sensitivity across datasets.
# =============================================================================
if step_enabled exp1; then
    for PAIR in "cifar100_resnet56:resnet56" "cifar100_resnet56:resnet20" "cifar100_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar100_}"
        for T in 1  5 10; do
            TAG="c100_${SHORT_T}_${STUDENT}_T${T}"
            add_job "${TAG}" $(common_args "${TEACHER}" "${STUDENT}" "${T}" "${RW}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed 42")
        done
    done
fi


# =============================================================================
# EXP 2 — Parameter Transfer: rw sweep on CIFAR-100
# =============================================================================
#   Do CIFAR-10 optimal parameters transfer to CIFAR-100?
#   rw = {0.05, 0.10, 0.20}
#   Fixed: d=5.0, beta=0.5, T=5, resnet56→resnet56
#   → 3 jobs
# =============================================================================
if step_enabled exp2; then
    for RW_VAL in 0.05 0.10 0.20; do
        TAG="c100_rw_sweep_${RW_VAL}"
        add_job "${TAG}" $(common_args "cifar100_resnet56" "resnet56" "${SWEEP_T}" "${RW_VAL}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 3 — Parameter Sensitivity: delta_logit sweep on CIFAR-100
# =============================================================================
#   100 classes dilute watermark signal by ~10x vs CIFAR-10.
#   Need to test stronger delta to compensate.
#   d = {3.0, 5.0, 7.0, 10.0}
#   Fixed: rw=0.1, beta=0.5, T=5, resnet56→resnet56
#   → 4 jobs
# =============================================================================
if step_enabled exp3; then
    for D_VAL in 3.0 5.0 7.0 10.0; do
        TAG="c100_d_sweep_${D_VAL}"
        add_job "${TAG}" $(common_args "cifar100_resnet56" "resnet56" "${SWEEP_T}" "${RW}" "${D_VAL}" "${BETA}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 4 — False Positive Rate (hard-label baseline)
# =============================================================================
#   Hard-label extraction → no watermark signal.
#   Expected: p >> eta.
#   → 2 jobs
# =============================================================================
if step_enabled exp4; then
    for PAIR in "cifar100_resnet56:resnet56" "cifar100_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        TAG="c100_fp_hard_${STUDENT}"
        add_job "${TAG}" \
            "--experiment surrogate_ft \
--model ${TEACHER} --student_arch ${STUDENT} \
--label_mode hard --trigger_mode rec_trigger \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} --temperatures ${SWEEP_T} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${TAG} --seed 42"
    done
fi


# =============================================================================
# Print summary & launch
# =============================================================================
echo ""
echo "============================================================"
echo " EvalGuard — CIFAR-100 Paper Experiments"
echo " GPUs       : ${GPU_LIST[*]} (${NUM_GPUS})"
echo " STEPS      : ${STEPS}"
echo " Total jobs : ${JOB_COUNT}"
echo " Jobs/GPU   : ~$(( (JOB_COUNT + NUM_GPUS - 1) / NUM_GPUS ))"
echo " LOG_DIR    : ${LOG_DIR}"
echo " Params     : DIST_EP=${DIST_EP}, NQ=${NQ}, RW=${RW}, D=${D_LOGIT}, BETA=${BETA}"
echo "============================================================"

echo ""
echo "GPU assignment (round-robin):"
for gpu_slot in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id="${GPU_LIST[$gpu_slot]}"
    count=0; tags=""
    while IFS=$'\t' read -r idx tag _; do
        if [ $((idx % NUM_GPUS)) -eq "${gpu_slot}" ]; then
            count=$((count + 1))
            tags="${tags} ${tag}"
        fi
    done < "${JOB_FILE}"
    echo "  GPU ${gpu_id} (${count} jobs):${tags}"
done

if [ "${DRY_RUN}" = "1" ]; then
    echo ""
    echo "[DRY RUN] Full job list:"
    while IFS=$'\t' read -r idx tag args; do
        gpu_id="${GPU_LIST[$((idx % NUM_GPUS))]}"
        echo "  [${idx}] GPU${gpu_id}  ${tag}"
        echo "       python experiments.py ${args}" | head -c 160
        echo ""
    done < "${JOB_FILE}"
    rm -f "${JOB_FILE}"
    exit 0
fi

# ── Launch per-GPU workers ──────────────────────────────────────
WORKER_PIDS=()
FAIL_FILE=$(mktemp /tmp/evalguard_c100_fails.XXXXXX)

for gpu_slot in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id="${GPU_LIST[$gpu_slot]}"
    (
        total=0
        while IFS=$'\t' read -r idx _ _; do
            [ $((idx % NUM_GPUS)) -eq "${gpu_slot}" ] && total=$((total + 1))
        done < "${JOB_FILE}"

        n=0
        while IFS=$'\t' read -r idx tag args; do
            [ $((idx % NUM_GPUS)) -ne "${gpu_slot}" ] && continue
            n=$((n + 1))
            echo ""
            echo "################################################################"
            echo "# [GPU${gpu_id}] (${n}/${total}) ${tag}"
            echo "# $(date '+%Y-%m-%d %H:%M:%S')"
            echo "################################################################"
            if CUDA_VISIBLE_DEVICES="${gpu_id}" python experiments.py ${args} 2>&1 | \
                    tee "${LOG_DIR}/${tag}.log"; then
                echo ">>> [GPU${gpu_id}] OK: ${tag}  $(date +%H:%M:%S)"
            else
                echo ">>> [GPU${gpu_id}] FAIL: ${tag}  $(date +%H:%M:%S)"
                echo "${tag}" >> "${FAIL_FILE}"
            fi
        done < "${JOB_FILE}"
    ) &
    WORKER_PIDS+=($!)
    echo "[launcher] GPU ${gpu_id} worker started (PID $!)"
done

echo ""
echo "[launcher] ${NUM_GPUS} GPU workers running. Logs: ${LOG_DIR}/"
echo ""

for pid in "${WORKER_PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
rm -f "${JOB_FILE}"

echo ""
echo "============================================================"
echo " DONE — CIFAR-100  $(date '+%Y-%m-%d %H:%M:%S')"
echo " Logs: ${LOG_DIR}/"
if [ -s "${FAIL_FILE}" ]; then
    n_fail=$(wc -l < "${FAIL_FILE}")
    echo " FAILED (${n_fail}):"
    while read -r t; do echo "   - ${t}"; done < "${FAIL_FILE}"
else
    echo " All ${JOB_COUNT} jobs succeeded."
fi
rm -f "${FAIL_FILE}"
echo "============================================================"