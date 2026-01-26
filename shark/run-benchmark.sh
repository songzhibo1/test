#!/bin/bash
set -e

# Default values
THREADS=4
MODE="malicious"
BENCHMARKS="simc1"

# Help message
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --threads NUM      Set OMP_NUM_THREADS (default: 4)"
    echo "  -m, --mode MODE        Set mode: 'malicious' or 'semi-honest' (default: malicious)"
    echo "  -b, --benchmarks LIST  Comma-separated list of benchmarks (default: simc1)"
    echo "  -a, --all              Run all benchmarks"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -t 4 -m semi-honest -b simc1"
    echo "  $0 --threads 8 --mode malicious --benchmarks simc1,simc2,hinet"
    echo "  $0 -t 4 -m semi-honest -a"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -b|--benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        -a|--all)
            BENCHMARKS="deepsecure1,deepsecure2,deepsecure3,deepsecure4,simc1,simc2,hinet,alexnet,vgg16,bert"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Set semi-honest flag
if [ "$MODE" = "semi-honest" ] || [ "$MODE" = "sh" ]; then
    SH_FLAG="--semi-honest"
    MODE_DISPLAY="Semi-honest"
else
    SH_FLAG=""
    MODE_DISPLAY="Malicious"
fi

# Convert comma-separated to space-separated
BENCHMARKS=$(echo "$BENCHMARKS" | tr ',' ' ')

echo "========================================"
echo "Shark Benchmark Runner"
echo "========================================"
echo "Mode:        $MODE_DISPLAY"
echo "Threads:     $THREADS"
echo "Benchmarks:  $BENCHMARKS"
echo "========================================"
echo ""

# Run benchmarks
for benchmark in $BENCHMARKS; do
    echo "----------------------------------------"
    echo "Benchmarking: $benchmark"
    echo "Mode: $MODE_DISPLAY | Threads: $THREADS"
    echo "----------------------------------------"

    # Generate keys
    ./build/benchmark-$benchmark 2 $SH_FLAG &> /dev/null

    # Run both parties
    OMP_NUM_THREADS=$THREADS ./build/benchmark-$benchmark 0 127.0.0.1 $SH_FLAG &> tmp0.txt &
    PID0=$!
    sleep 0.5
    OMP_NUM_THREADS=$THREADS ./build/benchmark-$benchmark 1 127.0.0.1 $SH_FLAG &> tmp1.txt
    wait $PID0

    echo ""
    echo "Party 0:"
    cat tmp0.txt
    echo ""
    echo "Party 1:"
    cat tmp1.txt
    echo ""

    rm -f tmp0.txt tmp1.txt
done

echo "========================================"
echo "Benchmark completed!"
echo "========================================"
