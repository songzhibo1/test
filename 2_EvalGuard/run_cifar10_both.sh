#!/bin/bash
# EvalGuard CIFAR-10: rec_trigger vs own_trigger at multiple trigger sizes
set -e

DEVICE="cuda"
DEVICE="cpu"
SCRIPT="python experiments.py"

NQ=50000
DIST_EP=80
RW=0.1
TEMPS="5,10"
D_LOGIT=5.0
BETA=0.7
V_TEMP=10.0

# ============================================================
# Trigger size sweep: both rec and own use the same size per run
# 0 = use all available triggers
# ============================================================
TRIGGER_SIZES=(50000)

echo "========================================================"
echo "========================================================"
echo " CIFAR-10: trigger size sweep"
echo " Sizes: ${TRIGGER_SIZES[*]}"
echo "========================================================"

for TS in "${TRIGGER_SIZES[@]}"; do
    echo ""
    echo "========================================================"
    echo " Trigger size = ${TS} (0=all)"
    echo "========================================================"

    # [1] VGG-11 -> VGG-11
    echo ">>> VGG-11 -> VGG-11, trigger_size=${TS}"
    $SCRIPT --experiment surrogate_ft --model cifar10_vgg11 --student_arch vgg11 \
        --label_mode soft --trigger_mode own_trigger \
        --rec_trigger_size $TS --own_trigger_size $TS \
        --rw $RW --delta_logit $D_LOGIT --beta $BETA --verify_temperature $V_TEMP \
        --temperatures "$TEMPS" --nq $NQ --dist_epochs $DIST_EP --device $DEVICE

    # [2] ResNet-20 -> ResNet-20
    echo ">>> ResNet-20 -> ResNet-20, trigger_size=${TS}"
    $SCRIPT --experiment surrogate_ft --model cifar10_resnet20 --student_arch resnet20 \
        --label_mode soft --trigger_mode own_trigger \
        --rec_trigger_size $TS --own_trigger_size $TS \
        --rw $RW --delta_logit $D_LOGIT --beta $BETA --verify_temperature $V_TEMP \
        --temperatures "$TEMPS" --nq $NQ --dist_epochs $DIST_EP --device $DEVICE

    # [3] ResNet-20 -> VGG-11 (cross-architecture)
    echo ">>> ResNet-20 -> VGG-11 (cross-arch), trigger_size=${TS}"
    $SCRIPT --experiment surrogate_ft --model cifar10_resnet20 --student_arch vgg11 \
        --label_mode soft --trigger_mode own_trigger \
        --rec_trigger_size $TS --own_trigger_size $TS \
        --rw $RW --delta_logit $D_LOGIT --beta $BETA --verify_temperature $V_TEMP \
        --temperatures "$TEMPS" --nq $NQ --dist_epochs $DIST_EP --device $DEVICE
done

echo ""
echo "CIFAR-10 trigger sweep DONE!"
echo "Results in: results/surrogate_ft/cifar10/"