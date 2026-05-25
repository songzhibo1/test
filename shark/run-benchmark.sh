#!/bin/bash
set -e

# Default values
THREADS="default"
MODE="both"
BENCHMARKS="simc1"

# Help message
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --threads NUM      Set OMP_NUM_THREADS (default: system default, use 0 or 'default' for no limit)"
    echo "  -m, --mode MODE        Set mode: 'malicious', 'semi-honest', or 'both' (default: both)"
    echo "  -b, --benchmarks LIST  Comma-separated list of benchmarks (default: simc1)"
    echo "  -a, --all              Run all benchmarks"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -b simc1                           # Run both modes with system default threads"
    echo "  $0 -t 4 -b simc1                      # Run both modes with 4 threads"
    echo "  $0 -t 0 -b simc1                      # Run with no thread limit (system default)"
    echo "  $0 -t 8 -m semi-honest -b simc1      # Run only semi-honest mode with 8 threads"
    echo "  $0 -t 4 -a                            # Run all benchmarks in both modes"
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

# Convert comma-separated to space-separated
BENCHMARKS=$(echo "$BENCHMARKS" | tr ',' ' ')

echo "========================================"
echo "Shark Benchmark Runner"
echo "========================================"
echo "Threads:     $THREADS"
echo "Mode:        $MODE"
echo "Benchmarks:  $BENCHMARKS"
echo "========================================"
echo ""

# Function to run a single benchmark with a specific mode
run_benchmark() {
    local benchmark=$1
    local mode=$2
    local sh_flag=""

    if [ "$mode" = "semi-honest" ]; then
        sh_flag="--semi-honest"
    fi

    # Generate keys
    ./build/benchmark-$benchmark 2 $sh_flag &> /dev/null

    # Run both parties (with or without OMP_NUM_THREADS)
    if [ "$THREADS" = "default" ] || [ "$THREADS" = "0" ]; then
        # No thread limit - use system default
        ./build/benchmark-$benchmark 0 127.0.0.1 $sh_flag &> tmp0.txt &
        PID0=$!
        sleep 0.5
        ./build/benchmark-$benchmark 1 127.0.0.1 $sh_flag &> tmp1.txt
    else
        OMP_NUM_THREADS=$THREADS ./build/benchmark-$benchmark 0 127.0.0.1 $sh_flag &> tmp0.txt &
        PID0=$!
        sleep 0.5
        OMP_NUM_THREADS=$THREADS ./build/benchmark-$benchmark 1 127.0.0.1 $sh_flag &> tmp1.txt
    fi
    wait $PID0

    # Extract results from Party 1 (has the timing info)
    local time=$(grep "^${benchmark}:" tmp1.txt | awk '{print $2}')
    local comm=$(grep "^${benchmark}:" tmp1.txt | awk '{print $4}')
    local reconstruct=$(grep "^reconstruct:" tmp1.txt | awk '{print $2}')
    local reconstruct_comm=$(grep "^reconstruct:" tmp1.txt | awk '{print $4}')

    rm -f tmp0.txt tmp1.txt

    echo "$time $comm $reconstruct $reconstruct_comm"
}

# Run benchmarks
for benchmark in $BENCHMARKS; do
    echo "========================================"
    echo "Benchmark: $benchmark | Threads: $THREADS"
    echo "========================================"

    if [ "$MODE" = "both" ]; then
        # Run malicious mode
        echo -n "Running malicious mode...    "
        result_mal=$(run_benchmark $benchmark "malicious")
        time_mal=$(echo $result_mal | awk '{print $1}')
        comm_mal=$(echo $result_mal | awk '{print $2}')
        recon_mal=$(echo $result_mal | awk '{print $3}')

        # Run semi-honest mode
        echo -n "Running semi-honest mode...  "
        result_sh=$(run_benchmark $benchmark "semi-honest")
        time_sh=$(echo $result_sh | awk '{print $1}')
        comm_sh=$(echo $result_sh | awk '{print $2}')
        recon_sh=$(echo $result_sh | awk '{print $3}')

        echo "Done!"
        echo ""
        echo "----------------------------------------"
        printf "| %-12s | %10s | %10s |\n" "Mode" "Time" "Comm"
        echo "----------------------------------------"
        printf "| %-12s | %10s | %10s |\n" "Malicious" "$time_mal" "$comm_mal"
        printf "| %-12s | %10s | %10s |\n" "Semi-honest" "$time_sh" "$comm_sh"
        echo "----------------------------------------"

        # Calculate improvement if both values are numbers
        if [[ "$time_mal" =~ ^[0-9]+$ ]] && [[ "$time_sh" =~ ^[0-9]+$ ]]; then
            time_diff=$((time_mal - time_sh))
            time_pct=$(echo "scale=1; $time_diff * 100 / $time_mal" | bc)
            echo "Time improvement: ${time_diff} ms (${time_pct}%)"
        fi

        if [[ "$comm_mal" =~ ^[0-9.]+$ ]] && [[ "$comm_sh" =~ ^[0-9.]+$ ]]; then
            comm_pct=$(echo "scale=1; ($comm_mal - $comm_sh) * 100 / $comm_mal" | bc)
            echo "Comm reduction: ${comm_pct}%"
        fi

    elif [ "$MODE" = "semi-honest" ] || [ "$MODE" = "sh" ]; then
        echo "Running semi-honest mode..."
        run_benchmark $benchmark "semi-honest" > /dev/null
        echo ""
        echo "Party 1 results:"
        cat tmp1.txt 2>/dev/null || true

    else
        echo "Running malicious mode..."
        run_benchmark $benchmark "malicious" > /dev/null
        echo ""
        echo "Party 1 results:"
        cat tmp1.txt 2>/dev/null || true
    fi

    echo ""
done

echo "========================================"
echo "Benchmark completed!"
echo "========================================"
