#!/bin/bash
# =============================================================================
# EvalGuard — diagnostic sweep
# =============================================================================
#
# Runs the experiments you need to answer:
#   1. Is the T=5 "negative shift" a true absence of watermark signal,
#      or an artefact of the single-control-class Wilcoxon design?
#   2. What delta_logit / beta combination restores T=5?
#   3. Does own_trigger degrade when using testset (true zero-leakage)
#      instead of trainset?
#
# The code changes that back this script (already committed):
#   - watermark.py   : verify_ownership_all_designs runs THREE control
#                      designs (single_ctrl, mean_rest, suspect_top1)
#                      on the same forward pass and records all three
#                      results in the output JSON under 'all_designs'.
#   - experiments.py : --own_data_source {trainset,testset} switch
#                      for own_trigger reconstruction.
#
# Multi-GPU
# ---------
# Set NUM_GPUS=N to dispatch independent runs round-robin across N GPUs
# via CUDA_VISIBLE_DEVICES. Each experiments.py process is single-GPU;
# we parallelize by running up to NUM_GPUS processes concurrently.
# (This is faster than nn.DataParallel for small CIFAR-10 models because
# scaling efficiency on resnet20 is poor — the overhead of splitting each
# batch dominates. For larger models you can wrap the student in
# nn.DataParallel, but it is not needed here.)
#
# Usage examples:
#   ./run_diagnostics.sh                  # single GPU, full sweep
#   NUM_GPUS=4 ./run_diagnostics.sh       # 4 GPUs round-robin
#   NUM_GPUS=4 STEPS="repro scan_d" ./run_diagnostics.sh   # subset
#   NUM_GPUS=4 DRY_RUN=1 ./run_diagnostics.sh              # print only
# =============================================================================

set -e
cd "$(dirname "$0")"

# -----------------------------------------------------------------------------
# Tunables
# -----------------------------------------------------------------------------
#   NUM_GPUS     : number of concurrent jobs (one job per physical GPU).
#   GPU_IDS      : comma-separated physical GPU IDs to use. Overrides the
#                  default 0..NUM_GPUS-1 assignment.
#   SPLIT_TEMPS  : when =1, a job that would otherwise call experiments.py
#                  with --temperatures "1,3,5,10" is split into four separate
#                  single-T jobs.
#   STEPS        : which steps to run (space-separated subset of
#                  "repro scan_d own_testset final").
#   DRY_RUN      : print commands without executing.
# -----------------------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS="${GPU_IDS:-}"
SPLIT_TEMPS="${SPLIT_TEMPS:-0}"
DRY_RUN="${DRY_RUN:-0}"
STEPS="${STEPS:-repro scan_d own_testset final}"   # space-separated

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

MODEL_PAIR="${MODEL_PAIR:-cifar10_resnet20:resnet20}"
NQ="${NQ:-50000}"
DIST_EP="${DIST_EP:-80}"
FT_EP="${FT_EP:-20}"

# Build the concrete GPU list: either from GPU_IDS or from 0..NUM_GPUS-1
if [ -n "${GPU_IDS}" ]; then
    IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
    NUM_GPUS="${#GPU_LIST[@]}"
else
    GPU_LIST=()
    for _i in $(seq 0 $(( NUM_GPUS - 1 ))); do
        GPU_LIST+=("${_i}")
    done
fi

echo "============================================================"
echo " EvalGuard diagnostic sweep"
echo " NUM_GPUS    = ${NUM_GPUS}"
echo " GPU_LIST    = ${GPU_LIST[*]}"
echo " SPLIT_TEMPS = ${SPLIT_TEMPS}"
echo " STEPS       = ${STEPS}"
echo " MODEL       = ${MODEL_PAIR}"
echo " NQ=${NQ} DIST_EP=${DIST_EP} FT_EP=${FT_EP}"
echo " LOG_DIR     = ${LOG_DIR}"
echo " DRY_RUN     = ${DRY_RUN}"
echo "============================================================"

# -----------------------------------------------------------------------------
# Dataset & Model pre-download guard (prevents races when NUM_GPUS>1)
# -----------------------------------------------------------------------------
if [ "${DRY_RUN}" != "1" ]; then
    echo ""
    echo "[prep] ensuring Dataset and torch.hub models are downloaded before parallel workers..."
    export PREP_TEACHER="${MODEL_PAIR%%:*}"
    export PREP_STUDENT="${MODEL_PAIR##*:}"
    python - <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from evalguard.configs import CONFIGS, create_student

teacher_name = os.environ.get("PREP_TEACHER", "cifar10_resnet20")
student_arch = os.environ.get("PREP_STUDENT", "resnet20")

# 1. Cache Dataset
if teacher_name in CONFIGS:
    ds_fn = CONFIGS[teacher_name]["data_fn"]
    if ds_fn:
        _ = ds_fn()
        print("  Dataset ready.")

# 2. Cache Teacher Model
if teacher_name in CONFIGS:
    model_fn = CONFIGS[teacher_name]["model_fn"]
    try:
        _ = model_fn(pretrained=True)
        print(f"  Teacher ({teacher_name}) ready.")
    except Exception as e:
        print(f"  [WARN] Failed to load teacher: {e}")

# 3. Cache Student Model
num_classes = CONFIGS[teacher_name]["num_classes"] if teacher_name in CONFIGS else 10
try:
    _ = create_student(num_classes=num_classes, arch=student_arch)
    print(f"  Student ({student_arch}) ready.")
except Exception as e:
    print(f"  [WARN] Failed to load student: {e}")

print("  Pre-caching complete.")
PY
fi

# -----------------------------------------------------------------------------
# Job dispatcher
# -----------------------------------------------------------------------------
JOB_INDEX=0
WAVE_PIDS=()

run_job() {
    local tag="$1"; shift
    local rw="$1"; shift
    local d_logit="$1"; shift
    local beta="$1"; shift
    local v_temp="$1"; shift
    local temps="$1"; shift
    local trigger_mode="$1"; shift
    local own_src="$1"; shift
    local extra="$*"

    local gpu="${GPU_LIST[$(( JOB_INDEX % NUM_GPUS ))]}"
    local log_file="${LOG_DIR}/${tag}.log"

    local teacher="${MODEL_PAIR%%:*}"
    local student="${MODEL_PAIR##*:}"

    local cmd="CUDA_VISIBLE_DEVICES=${gpu} python experiments.py \
        --experiment surrogate_ft \
        --model ${teacher} --student_arch ${student} \
        --label_mode soft \
        --trigger_mode ${trigger_mode} \
        --own_data_source ${own_src} \
        --rw ${rw} --delta_logit ${d_logit} --beta ${beta} --delta_min 0.5 \
        --verify_temperature ${v_temp} \
        --temperatures ${temps} \
        --nq ${NQ} \
        --dist_epochs ${DIST_EP} --dist_lr 0.002 --dist_batch 128 \
        --ft_epochs ${FT_EP} --ft_lr 0.0005 --ft_fractions 0.0,0.01,0.05,0.10 \
        --device cuda \
        --tag ${tag} \
        ${extra}"

    echo ""
    echo "[$(date +%H:%M:%S)] GPU ${gpu}  ${tag}  (log: ${log_file})"
    echo "  ${cmd}"

    if [ "${DRY_RUN}" = "1" ]; then
        JOB_INDEX=$(( JOB_INDEX + 1 ))
        return 0
    fi

    ( eval "${cmd}" > "${log_file}" 2>&1 ) &
    WAVE_PIDS+=($!)
    JOB_INDEX=$(( JOB_INDEX + 1 ))

    # Fill the wave; once NUM_GPUS jobs are queued, wait for all of them
    if [ ${#WAVE_PIDS[@]} -ge ${NUM_GPUS} ]; then
        wave_wait
    fi
}

wave_wait() {
    local rc=0
    for pid in "${WAVE_PIDS[@]}"; do
        if ! wait "${pid}"; then
            rc=1
        fi
    done
    WAVE_PIDS=()
    if [ ${rc} -ne 0 ]; then
        echo ""
        echo "[FAIL] one or more jobs in the last wave exited non-zero — see logs."
        exit 1
    fi
}

step_enabled() {
    local s="$1"
    for x in ${STEPS}; do [ "$x" = "$s" ] && return 0; done
    return 1
}

run_job_temps() {
    local tag="$1"
    local rw="$2"
    local d_logit="$3"
    local beta="$4"
    local v_temp="$5"
    local temps="$6"
    local trigger_mode="$7"
    local own_src="$8"

    if [ "${SPLIT_TEMPS}" = "1" ] && [[ "${temps}" == *","* ]]; then
        IFS=',' read -ra _TARR <<< "${temps}"
        for _t in "${_TARR[@]}"; do
            run_job "${tag}_T${_t}" "${rw}" "${d_logit}" "${beta}" \
                    "${v_temp}" "${_t}" "${trigger_mode}" "${own_src}"
        done
    else
        run_job "${tag}" "${rw}" "${d_logit}" "${beta}" \
                "${v_temp}" "${temps}" "${trigger_mode}" "${own_src}"
    fi
}

# -----------------------------------------------------------------------------
# Step 1 — "repro"
# -----------------------------------------------------------------------------
if step_enabled repro; then
    echo ""
    echo "###################  STEP 1 — repro  ###################"
    run_job_temps "diag_repro"   0.05 2.0 0.3 5.0 "1,3,5,10" "both" "trainset"
fi

# -----------------------------------------------------------------------------
# Step 2 — "scan_d"
# -----------------------------------------------------------------------------
if step_enabled scan_d; then
    echo ""
    echo "###################  STEP 2 — scan_d  ###################"
    for D in 1.5 2.0 3.0 4.0 5.0 7.0; do
        run_job "diag_scan_d${D}" 0.05 "${D}" 0.5 5.0 "5" "rec_trigger" "trainset"
    done
fi

# -----------------------------------------------------------------------------
# Step 3 — "own_testset"
# -----------------------------------------------------------------------------
if step_enabled own_testset; then
    echo ""
    echo "###################  STEP 3 — own_testset  ###################"
    run_job_temps "diag_own_test" 0.05 2.0 0.3 5.0 "1,3,5" "own_trigger" "testset"
fi

# -----------------------------------------------------------------------------
# Step 4 — "final"
# -----------------------------------------------------------------------------
if step_enabled final; then
    echo ""
    echo "###################  STEP 4 — final  ###################"
    run_job_temps "diag_final"   0.1  5.0 0.5 10.0 "1,3,5,10" "both" "trainset"
fi

# Drain any remaining jobs
if [ ${#WAVE_PIDS[@]} -gt 0 ]; then
    wave_wait
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " DONE"
echo " Logs:    ${LOG_DIR}/*.log"
echo " Results: results/surrogate_ft/cifar10/.../soft_label__{rec,own}_trigger/"
echo "============================================================"
echo ""
echo "Quick scan of diagnostic designs in every 'diag_*' result:"
echo ""

if [ "${DRY_RUN}" != "1" ]; then
    python - <<'PY'
import glob, json, os, re

paths = sorted(glob.glob("results/surrogate_ft/cifar10/**/soft_label__*/*diag_*.json",
                         recursive=True))
if not paths:
    print("  (no diag_* results found yet)")
else:
    hdr = "{:<52s} {:<13s} {:<12s} {:>+9s} {:>+9s} {:>9s} {:>6s}".format(
        "file", "trig_mode", "design", "shift", "median", "p", "V")
    print(hdr)
    print("-" * len(hdr))
    for p in paths:
        d = json.load(open(p))
        tag_m = re.search(r"tag_([^_.]+(?:_[^_.]+)*)", os.path.basename(p))
        ft0 = (d.get("ft_results") or [{}])[0]
        ad = ft0.get("all_designs") or {}
        short = os.path.basename(p)[:50]
        trig = d.get("trigger_mode", "?")
        if not ad:
            print("{:<52s} {:<13s} {:<12s} {:>+9.5f} {:>9s} {:>9.2e} {:>6}".format(
                short, trig, "(legacy)",
                ft0.get("confidence_shift", 0.0), "-",
                ft0.get("p_value", 1.0), str(ft0.get("verified", False))))
            continue
        for k in ("single_ctrl", "mean_rest", "suspect_top1"):
            r = ad.get(k) or {}
            print("{:<52s} {:<13s} {:<12s} {:>+9.5f} {:>+9.5f} {:>9.2e} {:>6}".format(
                short, trig, k,
                r.get("confidence_shift", 0.0),
                r.get("median_shift", 0.0),
                r.get("p_value", 1.0),
                str(r.get("verified", False))))
PY
fis