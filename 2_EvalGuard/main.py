"""
EvalGuard — Main Entry Point

Usage:
    python main.py                          # Run fidelity test (default)
    python main.py --experiment all         # Run all experiments
    python main.py --experiment distill     # Run distillation only
    python main.py --pretrained resnet18_cifar10.pth  # Use pretrained weights
"""

from experiments import main

if __name__ == "__main__":
    main()
