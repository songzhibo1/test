#!/bin/bash
# =============================================================================
# EvalGuard — CIFAR-10 Paper Experiments
# =============================================================================
#
# Per-GPU queue: jobs round-robin across GPUs, each GPU runs its queue
# sequentially, all GPUs run in parallel. Output → terminal + log file.
#
# Usage:
#   GPU_IDS="0,1,2,3,4,5" ./run_cifar10_paper.sh
#   GPU_IDS="0,1,2,3,4,5" STEPS="exp1 exp2" ./run_cifar10_paper.sh
#   GPU_IDS="0,1,2,3,4,5" DRY_RUN=1 ./run_cifar10_paper.sh
#
# Available STEPS:
#   exp1  Main table (Table VII): 3 pairs × 4 temps               12 jobs
#   exp2  rw sweep:          {0.01, 0.02, 0.05, 0.10, 0.20}       5 jobs
#   exp3  delta_logit sweep: {1.0, 2.0, 3.0, 5.0, 7.0}            5 jobs
#   exp4  beta sweep:        {0.1, 0.3, 0.5, 0.7, 0.9}            5 jobs
#   exp5  nq (query budget): {5000, 10000, 20000, 50000}           4 jobs
#   exp6  trigger_size:      {100, 500, 1000, 2000, 0=all}         5 jobs
#   exp7  Multi-seed:        5 seeds                                5 jobs
#   exp8  False positive:    hard-label baseline                    2 jobs
#   exp9  Cross-arch:        vgg11→resnet20, resnet20→mobilenetv2   4 jobs
#                                                          Total = 47 jobs
# =============================================================================

cd "$(dirname "$0")"

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
GPU_IDS="${GPU_IDS:-0}"
STEPS="${STEPS:-exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 exp9}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs_cifar10}"
mkdir -p "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_LIST[@]}

# ─────────────────────────────────────────────────────────────────
# CIFAR-10 hyper-parameters
# ─────────────────────────────────────────────────────────────────
NQ=50000
DIST_EP=80            # 80 epochs sufficient for CIFAR-10 (10 classes)
DIST_LR=0.002
DIST_BATCH=128
FT_EP=20
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

JOB_FILE=$(mktemp /tmp/evalguard_c10_jobs.XXXXXX)
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
--label_mode soft --trigger_mode both --own_data_source trainset \
--rw ${rw} --delta_logit ${d} --beta ${beta} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} \
--temperatures ${temps} --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${tag} ${extra}"
}

# ─────────────────────────────────────────────────────────────────
# Pre-download guard (avoid torch.hub race condition under parallel)
# ─────────────────────────────────────────────────────────────────
if [ "${DRY_RUN}" != "1" ]; then
    echo "[prep] warming CIFAR-10 cache..."
    python - <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from evalguard.configs import cifar10_data, cifar10_resnet20, cifar10_vgg11, create_student
_ = cifar10_data();  print("  CIFAR-10 data ready.")
for fn in [cifar10_resnet20, cifar10_vgg11]:
    _ = fn(pretrained=True)
for arch in ["resnet20", "vgg11"]:
    _ = create_student(num_classes=10, arch=arch)
print("  Models cached.")
PY
fi

export TORCH_HUB_OFFLINE=1

# =============================================================================
# EXP 1 — Table VII: CIFAR-10 main results
# =============================================================================
#   3 architecture pairs × 4 temperatures → 12 jobs
#   Pairs: resnet20→resnet20, resnet20→vgg11, vgg11→vgg11
#   T = {1, 3, 5, 10}
# =============================================================================
if step_enabled exp1; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_resnet20:vgg11" "cifar10_vgg11:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        for T in 1 3 5 10; do
            TAG="c10_${SHORT_T}_${STUDENT}_T${T}"
            add_job "${TAG}" $(common_args "${TEACHER}" "${STUDENT}" "${T}" "${RW}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed 42")
        done
    done
fi


# =============================================================================
# EXP 2 — Parameter Sensitivity: rw (watermark ratio) sweep
# =============================================================================
#   rw = {0.01, 0.02, 0.05, 0.10, 0.20}
#   Fixed: d=5.0, beta=0.5, T=5, resnet20→resnet20
#   → 5 jobs
# =============================================================================
if step_enabled exp2; then
    for RW_VAL in 0.01 0.02 0.05 0.10 0.20; do
        TAG="rw_sweep_${RW_VAL}"
        add_job "${TAG}" $(common_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW_VAL}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 3 — Parameter Sensitivity: delta_logit sweep
# =============================================================================
#   d = {1.0, 2.0, 3.0, 5.0, 7.0}
#   Fixed: rw=0.1, beta=0.5, T=5, resnet20→resnet20
#   → 5 jobs
# =============================================================================
if step_enabled exp3; then
    for D_VAL in 1.0 2.0 3.0 5.0 7.0; do
        TAG="d_sweep_${D_VAL}"
        add_job "${TAG}" $(common_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_VAL}" "${BETA}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 4 — Parameter Sensitivity: beta (safety factor) sweep
# =============================================================================
#   beta = {0.1, 0.3, 0.5, 0.7, 0.9}
#   Fixed: rw=0.1, d=5.0, T=5, resnet20→resnet20
#   → 5 jobs
# =============================================================================
if step_enabled exp4; then
    for B_VAL in 0.1 0.3 0.5 0.7 0.9; do
        TAG="beta_sweep_${B_VAL}"
        add_job "${TAG}" $(common_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_LOGIT}" "${B_VAL}" "${TAG}" "--seed 42")
    done
fi


# =============================================================================
# EXP 5 — Attacker Cost: nq (query budget) sweep
# =============================================================================
#   How many queries does the attacker need to steal the watermark?
#   nq = {5000, 10000, 20000, 50000}
#   Fixed: rw=0.1, d=5.0, beta=0.5, T=5, resnet20→resnet20
#   → 4 jobs
# =============================================================================
if step_enabled exp5; then
    _NQ_BAK="${NQ}"
    for NQ_VAL in 5000 10000 20000 50000; do
        NQ="${NQ_VAL}"
        TAG="nq_sweep_${NQ_VAL}"
        add_job "${TAG}" $(common_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed 42")
    done
    NQ="${_NQ_BAK}"
fi


# =============================================================================
# EXP 6 — Defense Cost: trigger_size (verification set) sweep
# =============================================================================
#   How many trigger samples does the owner need for reliable verification?
#   trigger_size = {100, 500, 1000, 2000, 0=all(~5000)}
#   Fixed: rw=0.1, d=5.0, beta=0.5, T=5, resnet20→resnet20
#
#   All jobs share checkpoint via same --tag (only first distills, rest cache).
#   --seed 42 ensures identical K_w if parallel GPUs race.
#   → 5 jobs
# =============================================================================
if step_enabled exp6; then
    for TS in 100 500 1000 2000 0; do
        TS_LABEL="${TS}"
        [ "${TS}" = "0" ] && TS_LABEL="all"
        add_job "trig_sweep_${TS_LABEL}" \
            $(common_args "cifar10_resnet20" "resnet20" "${SWEEP_T}" "${RW}" "${D_LOGIT}" "${BETA}" \
                          "trig_sweep" \
                          "--rec_trigger_size ${TS} --own_trigger_size ${TS} --seed 42")
    done
fi


# =============================================================================
# EXP 7 — Multi-seed Reproducibility
# =============================================================================
#   5 seeds, same config → report mean ± std of confidence_shift
#   seeds = {42, 123, 256, 777, 2024}
#   Fixed: rw=0.1, d=5.0, T=3,5, resnet20→resnet20
#   → 5 jobs
# =============================================================================
if step_enabled exp7; then
    for SEED in 42 123 256 777 2024; do
        TAG="seed_${SEED}"
        add_job "${TAG}" $(common_args "cifar10_resnet20" "resnet20" "3,5" "${RW}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed ${SEED}")
    done
fi


# =============================================================================
# EXP 8 — False Positive Rate (hard-label baseline)
# =============================================================================
#   Hard-label extraction bypasses watermark (argmax only).
#   Expected: p-value >> eta (no false detection).
#   → 2 jobs
# =============================================================================
if step_enabled exp8; then
    for PAIR in "cifar10_resnet20:resnet20" "cifar10_resnet20:vgg11"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        TAG="fp_hard_${STUDENT}"
        add_job "${TAG}" \
            "--experiment surrogate_ft \
--model ${TEACHER} --student_arch ${STUDENT} \
--label_mode hard --trigger_mode rec_trigger \
--rw ${RW} --delta_logit ${D_LOGIT} --beta ${BETA} --delta_min ${DELTA_MIN} \
--verify_temperature ${V_TEMP} --temperatures 3 --nq ${NQ} \
--dist_epochs ${DIST_EP} --dist_lr ${DIST_LR} --dist_batch ${DIST_BATCH} \
--ft_epochs ${FT_EP} --ft_lr ${FT_LR} --ft_fractions ${FT_FRACS} \
--device cuda --tag ${TAG} --seed 42"
    done
fi


# =============================================================================
# EXP 9 — Extended Cross-Architecture
# =============================================================================
#   vgg11→resnet20 (reverse), resnet20→mobilenetv2 (lightweight student)
#   T = {3, 5}
#   → 4 jobs
# =============================================================================
if step_enabled exp9; then
    for PAIR in "cifar10_vgg11:resnet20" "cifar10_resnet20:mobilenetv2"; do
        TEACHER="${PAIR%%:*}"; STUDENT="${PAIR##*:}"
        SHORT_T="${TEACHER#cifar10_}"
        for T in 3 5; do
            TAG="xarch_${SHORT_T}_${STUDENT}_T${T}"
            add_job "${TAG}" $(common_args "${TEACHER}" "${STUDENT}" "${T}" "${RW}" "${D_LOGIT}" "${BETA}" "${TAG}" "--seed 42")
        done
    done
fi


# =============================================================================
# Print summary & launch
# =============================================================================
echo ""
echo "============================================================"
echo " EvalGuard — CIFAR-10 Paper Experiments"
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
FAIL_FILE=$(mktemp /tmp/evalguard_c10_fails.XXXXXX)

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
echo " DONE — CIFAR-10  $(date '+%Y-%m-%d %H:%M:%S')"
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