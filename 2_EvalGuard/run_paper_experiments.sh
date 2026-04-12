#!/bin/bash
# =============================================================================
# EvalGuard — Run ALL paper experiments (CIFAR-10 + CIFAR-100)
# =============================================================================
#
# This is a convenience wrapper. For fine-grained control, use:
#   ./run_cifar10_paper.sh   (47 jobs, 9 experiment groups)
#   ./run_cifar100_paper.sh  (21 jobs, 4 experiment groups)
#
# Usage:
#   GPU_IDS="0,1,2,3,4,5" ./run_paper_experiments.sh
#   GPU_IDS="0,1,2,3,4,5" DRY_RUN=1 ./run_paper_experiments.sh
#
# Note: STEPS is NOT forwarded (each sub-script has its own step namespace).
# To select specific steps, run the sub-scripts directly.
# =============================================================================

cd "$(dirname "$0")"

echo "============================================================"
echo " Running CIFAR-10 experiments..."
echo "============================================================"
./run_cifar10_paper.sh

echo ""
echo "============================================================"
echo " Running CIFAR-100 experiments..."
echo "============================================================"
./run_cifar100_paper.sh

echo ""
echo "============================================================"
echo " ALL experiments complete."
echo "============================================================"