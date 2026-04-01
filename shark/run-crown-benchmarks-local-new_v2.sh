#!/bin/bash
set -e

# ==================== 网络条件配置 ====================
# 可选值: LAN, WAN_1ms_1Gbps, WAN_10ms_500Mbps, WAN_40ms_100Mbps, WAN_100ms_50Mbps
# 也可以通过命令行参数指定: ./run-crown-benchmarks-local-new.sh WAN_40ms_100Mbps
NETWORK_CONDITION="${1:-LAN}"

# 网络条件定义 (格式: "delay bandwidth")
declare -A NETWORK_CONFIGS
NETWORK_CONFIGS["LAN"]="0.05ms 10Gbit"                    # 数据中心内部 (RTT ~0.1ms)
# 排序逻辑：带宽从高到低 (600 -> 370)，延迟从低到高 (20ms -> 30ms)
NETWORK_CONFIGS["WAN_40ms_600Mbps"]="20ms 600Mbit"   # 性能优 (RTT 40ms)
NETWORK_CONFIGS["WAN_60ms_600Mbps"]="30ms 600Mbit"   # RTT 60ms
NETWORK_CONFIGS["WAN_40ms_370Mbps"]="20ms 370Mbit"   # 带宽降至370M
NETWORK_CONFIGS["WAN_60ms_370Mbps"]="30ms 370Mbit"   # 性能最差
# 验证网络条件
if [ -z "${NETWORK_CONFIGS[$NETWORK_CONDITION]}" ]; then
    echo "ERROR: Invalid NETWORK_CONDITION '$NETWORK_CONDITION'"
    echo "Available options: ${!NETWORK_CONFIGS[*]}"
    exit 1
fi

# 解析网络参数
NET_PARAMS=(${NETWORK_CONFIGS[$NETWORK_CONDITION]})
NET_DELAY="${NET_PARAMS[0]}"
NET_RATE="${NET_PARAMS[1]}"

# 设置网络条件 (需要sudo权限)
setup_network() {
    echo "Setting up network condition: $NETWORK_CONDITION"
    echo "  Delay: $NET_DELAY (one-way), Rate: $NET_RATE"

    # 清空现有规则
    sudo tc qdisc del dev lo root 2>/dev/null || true

    # 设置新规则
    sudo tc qdisc add dev lo root netem delay $NET_DELAY rate $NET_RATE

    echo "Network condition applied successfully!"
    echo ""
}

# 清理网络条件
cleanup_network() {
    echo "Cleaning up network conditions..."
    sudo tc qdisc del dev lo root 2>/dev/null || true
}

# 设置网络条件
setup_network

# 结果目录前缀 (包含网络条件)
RESULTS_BASE_DIR="crown-results/${NETWORK_CONDITION}"

# ==================== 运行配置 ====================
# 线程配置列表: 0 表示不设置OMP_NUM_THREADS(使用系统默认), 其他数字表示使用指定线程数
# Dealer预处理只执行一次，Server/Client会按此列表分别执行
THREADS_LIST=(4)

# 运行模式: "both", "malicious", "semi-honest"
RUN_MODE="semi-honest"

# 创建结果目录
mkdir -p "$RESULTS_BASE_DIR"

# ==================== 网络配置 ====================
IP_ADDRESS="127.0.0.1"

# ==================== 模型配置 ====================
# 格式: "num_layers hidden_dim input_dim output_dim"
declare -A MODEL_CONFIGS

# MNIST 模型 (input_dim=20, output_dim=10)
MODEL_CONFIGS["mnist_2layer_relu_20_best"]="2 20 784 10"
MODEL_CONFIGS["mnist_3layer_relu_20_best"]="3 20 784 10"

# MNIST 模型 (input_dim=784, output_dim=10)
MODEL_CONFIGS["vnncomp_mnist_3layer_relu_256_best"]="3 256 784 10"
MODEL_CONFIGS["vnncomp_mnist_5layer_relu_256_best"]="5 256 784 10"
MODEL_CONFIGS["vnncomp_mnist_7layer_relu_256_best"]="7 256 784 10"

# CIFAR 模型 (input_dim=3072, output_dim=10)
MODEL_CONFIGS["cifar_6layer_relu_2048_best"]="6 2048 3072 10"
MODEL_CONFIGS["eran_cifar_5layer_relu_100_best"]="5 100 3072 10"
MODEL_CONFIGS["eran_cifar_7layer_relu_100_best"]="7 100 3072 10"
MODEL_CONFIGS["eran_cifar_10layer_relu_200_best"]="10 200 3072 10"

# ==================== 测试样本配置 ====================
# 格式: "id,true_class,target_class" 每组一行，更直观
# 数据来源: correct = True 的样本

declare -A IMAGE_CONFIGS


IMAGE_CONFIGS["mnist_3layer_relu_20_best"]="
0,7,6
1,2,4
2,1,5
3,0,3
4,4,0
5,1,6
6,4,0
9,9,6
10,0,4
11,6,7
12,9,6
13,0,4
14,1,0
15,5,4
16,9,1
17,7,6
19,4,0
20,9,6
21,6,7
22,6,3
23,5,2
24,4,0
25,0,3
26,7,6
27,4,0
28,0,4
29,1,0
30,3,6
31,1,0
32,3,4
33,4,7
34,7,6
35,2,4
36,7,6
37,1,0
38,2,4
39,1,2
40,1,0
41,7,6
42,4,6
43,2,9
44,3,0
45,5,7
46,1,0
47,2,9
48,4,0
49,4,6
50,6,7
51,3,4
52,5,2
53,5,4
54,6,3
55,0,4
56,4,0
57,1,5
58,9,6
59,5,6
60,7,6
64,7,6
65,4,0
66,6,4
67,4,6
68,3,4
69,0,4
70,7,6
71,0,4
72,2,4
73,9,6
74,1,0
75,7,6
76,3,4
77,2,4
78,9,6
79,7,6
80,7,6
81,6,3
82,2,4
83,7,6
84,8,0
85,4,0
86,7,6
87,3,0
88,6,3
89,1,0
90,3,4
91,6,7
92,9,0
93,3,6
94,1,0
95,4,3
96,1,0
97,7,6
98,6,7
99,9,6

"

IMAGE_CONFIGS["mnist_2layer_relu_20_best"]="
0,7,4
1,2,4
2,1,6
3,0,1
4,4,1
5,1,6
6,4,1
7,9,6
8,5,1
9,9,1
10,0,4
11,6,7
12,9,1
13,0,4
14,1,0
15,5,7
16,9,1
17,7,4
18,3,6
19,4,1
20,9,6
21,6,7
22,6,1
23,5,7
24,4,1
25,0,1
26,7,6
27,4,1
28,0,4
29,1,0
30,3,6
31,1,6
32,3,6
33,4,1
34,7,6
35,2,4
36,7,6
37,1,0
38,2,4
39,1,5
40,1,9
41,7,6
42,4,5
43,2,9
44,3,6
45,5,2
46,1,6
47,2,9
48,4,1
49,4,5
50,6,9
51,3,6
52,5,4
53,5,7
54,6,9
55,0,4
56,4,1
57,1,6
58,9,1
59,5,6
60,7,4
62,9,1
63,3,6
64,7,5
65,4,0
66,6,9
67,4,5
68,3,6
69,0,4
70,7,6
71,0,4
72,2,4
73,9,6
74,1,0
75,7,6
76,3,6
77,2,4
78,9,6
79,7,6
80,7,2
81,6,1
82,2,5
83,7,1
84,8,1
85,4,1
86,7,6
87,3,6
88,6,1
89,1,5
90,3,6
91,6,9
92,9,6
93,3,6
94,1,0
95,4,1
96,1,6
97,7,5
98,6,9
99,9,1

"



IMAGE_CONFIGS["vnncomp_mnist_3layer_relu_256_best"]="
0,7,3
1,2,1
"
#2,1,2
#3,0,7
#4,4,0
#5,1,2
#6,4,7
#7,9,7
#8,5,3
#9,9,7
#10,0,7
#11,6,5
#12,9,7
#13,0,5
#14,1,8
#15,5,4
#16,9,7
#17,7,2
#18,3,7
#19,4,0
#20,9,7
#21,6,5
#22,6,5
#23,5,7
#24,4,7
#25,0,7
#26,7,2
#27,4,0
#28,0,7
#29,1,4
#30,3,4
#31,1,4
#32,3,4
#33,4,7
#34,7,1
#35,2,7
#36,7,9
#37,1,4
#38,2,7
#39,1,8
#40,1,4
#41,7,2
#42,4,0
#43,2,0
#44,3,4
#45,5,4
#46,1,3
#47,2,7
#48,4,0
#49,4,0
#50,6,4
#51,3,4
#52,5,7
#53,5,4
#54,6,5
#55,0,7
#56,4,0
#57,1,3
#58,9,4
#59,5,7
#60,7,2
#61,8,1
#62,9,7
#63,3,4
#64,7,2
#65,4,7
#66,6,4
#67,4,0
#68,3,4
#69,0,5
#70,7,2
#71,0,5
#72,2,1
#73,9,4
#74,1,4
#75,7,2
#76,3,4
#77,2,1
#78,9,0
#79,7,9
#80,7,1
#81,6,5
#82,2,7
#83,7,1
#84,8,1
#85,4,0
#86,7,2
#87,3,4
#88,6,5
#89,1,2
#90,3,4
#91,6,5
#92,9,7
#93,3,7
#94,1,8
#95,4,7
#96,1,6
#97,7,2
#98,6,5
#99,9,7
IMAGE_CONFIGS["vnncomp_mnist_5layer_relu_256_best"]="
20,9,8
21,6,9
"
#22,6,9

IMAGE_CONFIGS["vnncomp_mnist_7layer_relu_256_best"]="
    0,7,3
1,2,5
"
#2,1,5
#3,0,3
#4,4,2
#5,1,5
#6,4,3
#7,9,6
#8,5,2
#9,9,6
#10,0,3
#11,6,2
#12,9,2
#13,0,3
#14,1,5
#15,5,3
#16,9,2
#17,7,3
#18,3,9
#19,4,2
#20,9,2
#21,6,2
#22,6,2
#23,5,3
#24,4,3
#25,0,3
#26,7,3
#27,4,2
#28,0,3
#29,1,5
#30,3,2
#31,1,0
#32,3,2
#33,4,3
#34,7,3
#35,2,5
#36,7,3
#37,1,5
#38,2,5
#39,1,5
#40,1,0
#41,7,3
#42,4,2
#43,2,5
#44,3,2
#45,5,3
#46,1,5
#47,2,5
#48,4,2
#49,4,2
#50,6,2
#51,3,2
#52,5,3
#53,5,3
#54,6,2
#55,0,3
#56,4,2
#57,1,5
#58,9,6
#59,5,4
#60,7,5
#61,8,1
#62,9,2
#64,7,3
#65,4,3
#66,6,2
#67,4,2
#68,3,2
#69,0,3
#70,7,3
#71,0,3
#72,2,5
#73,9,6
#74,1,5
#75,7,5
#76,3,2
#77,2,5
#78,9,2
#79,7,3
#80,7,5
#81,6,2
#82,2,5
#83,7,5
#84,8,3
#85,4,2
#86,7,3
#87,3,2
#88,6,2
#89,1,5
#90,3,2
#91,6,2
#92,9,2
#93,3,2
#94,1,5
#95,4,2
#96,1,0
#97,7,3
#98,6,2
#99,9,6

IMAGE_CONFIGS["cifar_6layer_relu_2048_best"]="
    0,3,3
"

# eran_cifar_5layer_relu_100_best 的测试样本 (correct = True)
IMAGE_CONFIGS["eran_cifar_5layer_relu_100_best"]="
0,3,6
3,0,4
"
#6,1,4
#7,6,7
#9,1,6
#10,0,6
#11,9,5
#13,7,4
#14,9,8
#18,8,6
#19,6,1
#21,0,1
#23,9,3
#24,5,8
#25,2,8
#28,9,6
#29,6,8
#33,5,8
#34,9,8
#37,1,5
#38,9,5
#45,9,6
#47,9,8
#48,7,8
#49,6,8
#50,9,5
#54,8,6
#55,8,6
#56,7,1
#60,7,8
#61,3,1
#66,1,4
#67,2,9
#68,3,0
#72,8,5
#73,8,5
#76,9,8
#81,1,8
#83,7,1
#86,2,1
#88,8,6
#89,9,6
#92,8,6
#96,6,8
#99,7,8

IMAGE_CONFIGS["eran_cifar_7layer_relu_100_best"]="
    5,6,0
6,1,4
"
#7,6,5
#9,1,6
#10,0,6
#12,5,4
#13,7,6
#14,9,4
#18,8,3
#19,6,0
#20,7,6
#21,0,5
#23,9,4
#28,9,4
#29,6,8
#33,5,8
#34,9,4
#38,9,6
#45,9,4
#46,3,8
#47,9,6
#48,7,6
#50,9,4
#53,3,6
#55,8,3
#56,7,6
#60,7,3
#66,1,4
#68,3,9
#73,8,2
#74,0,3
#75,2,1
#76,9,4
#77,3,0
#78,3,7
#80,8,7
#81,1,4
#82,1,7
#86,2,7
#88,8,2
#89,9,4
#90,0,3
#92,8,7
#97,0,7
#98,0,1
#99,7,6

IMAGE_CONFIGS["eran_cifar_10layer_relu_200_best"]="
    1,8,4
6,1,4
"
#7,6,4
#8,3,8
#9,1,6
#10,0,4
#11,9,4
#13,7,6
#14,9,4
#18,8,4
#19,6,8
#23,9,4
#24,5,9
#26,4,0
#28,9,4
#29,6,4
#31,5,4
#32,4,5
#34,9,4
#37,1,2
#38,9,5
#39,5,0
#44,0,5
#45,9,5
#47,9,4
#48,7,4
#49,6,2
#50,9,4
#54,8,5
#55,8,5
#60,7,2
#65,2,4
#66,1,2
#67,2,4
#68,3,6
#72,8,7
#73,8,5
#74,0,5
#76,9,4
#78,3,4
#79,8,6
#81,1,6
#82,1,5
#86,2,5
#88,8,6
#89,9,4
#90,0,5
#92,8,4
#93,6,4
#97,0,5
#98,0,4
#99,7,6

MODELS_TO_TEST=(
#    "mnist_2layer_relu_20_best"
    "mnist_3layer_relu_20_best"
#    "vnncomp_mnist_3layer_relu_256_best"
#    "vnncomp_mnist_5layer_relu_256_best"
#    "vnncomp_mnist_7layer_relu_256_best"
#    "eran_cifar_5layer_relu_100_best"
#    "eran_cifar_7layer_relu_100_best"
#    "eran_cifar_10layer_relu_200_best"

)

#EPS_LIST=(0.005 0.01 0.025 0.05 0.08 0.1 0.2)
#EPS_LIST=(0.001 0.0015 0.0018 0.0022 0.0035)
#EPS_LIST=(0.001 0.0016 0.0019 0.0025)
#EPS_LIST=(0.0078)
#EPS_LIST=(0.015 0.045 0.1)
EPS_LIST=(0.03)
# ==================== 辅助函数 ====================
bytes_to_human() {
    local bytes=$1
    if [ $bytes -ge 1073741824 ]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1073741824}") GB"
    elif [ $bytes -ge 1048576 ]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1048576}") MB"
    elif [ $bytes -ge 1024 ]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1024}") KB"
    else
        echo "$bytes bytes"
    fi
}

# 清理残留进程的函数
cleanup_processes() {
    echo "Cleaning up any remaining benchmark processes..."
    pkill -f "benchmark-crown" 2>/dev/null || true
    sleep 2
}

# 脚本退出时清理 (进程和网络)
cleanup_all() {
    cleanup_processes
    cleanup_network
}
trap cleanup_all EXIT

# 确定要运行的模式列表
MODES_TO_RUN=()
if [ "$RUN_MODE" = "both" ]; then
    MODES_TO_RUN=("malicious" "semi-honest")
elif [ "$RUN_MODE" = "malicious" ]; then
    MODES_TO_RUN=("malicious")
elif [ "$RUN_MODE" = "semi-honest" ]; then
    MODES_TO_RUN=("semi-honest")
else
    echo "ERROR: Invalid RUN_MODE '$RUN_MODE'. Use 'both', 'malicious', or 'semi-honest'."
    exit 1
fi

echo "========================================"
echo "Crown Benchmark Runner"
echo "========================================"
echo "Network:      $NETWORK_CONDITION ($NET_DELAY delay, $NET_RATE rate)"
echo "Results Dir:  $RESULTS_BASE_DIR"
echo "Threads List: ${THREADS_LIST[*]} (0 = system default)"
echo "Run Mode:     $RUN_MODE"
echo "Modes:        ${MODES_TO_RUN[*]}"
echo "========================================"
echo ""

# ==================== 主循环 ====================
for model in "${MODELS_TO_TEST[@]}"; do
    config=(${MODEL_CONFIGS[$model]})
    NUM_LAYERS=${config[0]}
    HIDDEN_DIM=${config[1]}
    INPUT_DIM=${config[2]}
    OUTPUT_DIM=${config[3]}

    # 解析测试样本配置
    IMAGE_LIST=()
    while IFS= read -r line; do
        # 去除空白并跳过空行
        line=$(echo "$line" | tr -d ' ')
        if [ -n "$line" ]; then
            IMAGE_LIST+=("$line")
        fi
    done <<< "${IMAGE_CONFIGS[$model]}"

    echo ""
    echo "########################################################"
    echo "# Model: $model"
    echo "# Layers: $NUM_LAYERS, Hidden: $HIDDEN_DIM"
    echo "# Input: $INPUT_DIM, Output: $OUTPUT_DIM"
    echo "# Test samples: ${#IMAGE_LIST[@]}"
    echo "#"
    echo "# Sample configs (id,true_class,target_class):"
    for sample in "${IMAGE_LIST[@]}"; do
        echo "#   $sample"
    done
    echo "########################################################"
    echo ""

    mkdir -p $RESULTS_BASE_DIR/$model

    for eps in "${EPS_LIST[@]}"; do
        echo "=========================================="
        echo "Testing EPS = $eps"
        echo "=========================================="

        mkdir -p $RESULTS_BASE_DIR/$model/eps_$eps

        # 遍历所有测试样本
        for sample in "${IMAGE_LIST[@]}"; do
            # 解析 id,true_class,target_class
            IFS=',' read -r IMAGE_ID TRUE_LABEL TARGET_LABEL <<< "$sample"

            echo ""
            echo "------------------------------------------"
            echo "Image $IMAGE_ID: true_class=$TRUE_LABEL, target_class=$TARGET_LABEL"
            echo "------------------------------------------"

            # 遍历每种模式
            for current_mode in "${MODES_TO_RUN[@]}"; do
                # 设置模式标志和文件后缀
                if [ "$current_mode" = "semi-honest" ]; then
                    SH_FLAG="--semi-honest"
                    MODE_SUFFIX="_sh"
                    MODE_DISPLAY="Semi-Honest"
                else
                    SH_FLAG=""
                    MODE_SUFFIX="_m"
                    MODE_DISPLAY="Malicious"
                fi

                echo ""
                echo ">>> Running $MODE_DISPLAY mode..."

                # 构建自定义参数（不含线程相关）
                CUSTOM_ARGS="--model=$model --num_layers=$NUM_LAYERS --hidden_dim=$HIDDEN_DIM --input_dim=$INPUT_DIM --output_dim=$OUTPUT_DIM --eps=$eps --true_label=$TRUE_LABEL --target_label=$TARGET_LABEL --image_id=$IMAGE_ID $SH_FLAG"

                # ========== Dealer 预处理（每个模式只执行一次）==========
                # 使用标记文件来记录当前dat文件是哪个配置生成的
                # 标记格式: model_imageID_eps_mode (例如: eran_cifar_7layer_relu_100_best_5_0.001_sh)
                DAT_MODE_MARKER=".dat_mode_marker"
                CURRENT_CONFIG="${model}_${IMAGE_ID}_${eps}${MODE_SUFFIX}"

                # 检查是否已有当前模式的dealer结果
                DEALER_DONE=false

                # 检查dealer.txt是否存在（新格式或旧格式）
                DEALER_TXT_EXISTS=false
                for t in "${THREADS_LIST[@]}"; do
                    if [ -f "$RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}_t${t}_dealer.txt" ]; then
                        DEALER_TXT_EXISTS=true
                        break
                    fi
                done
                # 兼容检查无线程后缀的旧dealer文件
                if [ -f "$RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}_dealer.txt" ]; then
                    DEALER_TXT_EXISTS=true
                fi

                # 检查dat文件是否存在且是当前配置生成的
                DAT_FILES_VALID=false
                if [ -f "server.dat" ] && [ -f "client.dat" ]; then
                    if [ -f "$DAT_MODE_MARKER" ]; then
                        SAVED_CONFIG=$(cat "$DAT_MODE_MARKER")
                        if [ "$SAVED_CONFIG" = "$CURRENT_CONFIG" ]; then
                            DAT_FILES_VALID=true
                        else
                            echo "INFO: dat files are from different config ($SAVED_CONFIG), will regenerate for $CURRENT_CONFIG"
                        fi
                    else
                        echo "INFO: dat mode marker not found, will regenerate dat files"
                    fi
                fi

                # 只有当dealer.txt存在且dat文件有效时，才跳过预处理
                if [ "$DEALER_TXT_EXISTS" = true ] && [ "$DAT_FILES_VALID" = true ]; then
                    DEALER_DONE=true
                fi

                if [ "$DEALER_DONE" = false ]; then
                    # 清理之前的进程
                    cleanup_processes

                    # 记录 DEALER 开始时间
                    DEALER_START=$(date +%s.%N)
                    echo "[$(date +'%H:%M:%S')] Starting DEALER preprocessing ($MODE_DISPLAY)..."

                    # DEALER (party 2) - 预处理与线程无关，只执行一次
                    ./build/benchmark-crown 2 $CUSTOM_ARGS &> $RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}_dealer.txt
                    DEALER_EXIT_CODE=$?

                    DEALER_END=$(date +%s.%N)
                    DEALER_TIME=$(echo "$DEALER_END - $DEALER_START" | bc)

                    echo "[$(date +'%H:%M:%S')] DEALER preprocessing completed! (${DEALER_TIME}s)"

                    # 只有dealer成功且dat文件生成后才更新marker
                    if [ $DEALER_EXIT_CODE -eq 0 ] && [ -f "server.dat" ] && [ -f "client.dat" ]; then
                        echo "$CURRENT_CONFIG" > "$DAT_MODE_MARKER"
                        echo "Updated dat marker: $CURRENT_CONFIG"
                    else
                        echo "WARNING: Dealer may have failed (exit code: $DEALER_EXIT_CODE), marker not updated"
                    fi

                    # 获取文件大小
                    if [ -f "server.dat" ]; then
                        SERVER_SIZE=$(stat -c%s server.dat)
                        SERVER_HUMAN=$(bytes_to_human $SERVER_SIZE)
                    else
                        SERVER_SIZE=0
                        SERVER_HUMAN="NOT FOUND"
                    fi

                    if [ -f "client.dat" ]; then
                        CLIENT_SIZE=$(stat -c%s client.dat)
                        CLIENT_HUMAN=$(bytes_to_human $CLIENT_SIZE)
                    else
                        CLIENT_SIZE=0
                        CLIENT_HUMAN="NOT FOUND"
                    fi

                    TOTAL_SIZE=$((SERVER_SIZE + CLIENT_SIZE))
                    TOTAL_HUMAN=$(bytes_to_human $TOTAL_SIZE)

                    echo "Files: server.dat ($SERVER_HUMAN), client.dat ($CLIENT_HUMAN)"
                else
                    echo "[$(date +'%H:%M:%S')] DEALER preprocessing already done for $MODE_DISPLAY mode (eps=$eps), skipping..."
                    # 重新获取文件大小用于后续summary
                    if [ -f "server.dat" ]; then
                        SERVER_SIZE=$(stat -c%s server.dat)
                        SERVER_HUMAN=$(bytes_to_human $SERVER_SIZE)
                    else
                        SERVER_SIZE=0
                        SERVER_HUMAN="NOT FOUND"
                    fi
                    if [ -f "client.dat" ]; then
                        CLIENT_SIZE=$(stat -c%s client.dat)
                        CLIENT_HUMAN=$(bytes_to_human $CLIENT_SIZE)
                    else
                        CLIENT_SIZE=0
                        CLIENT_HUMAN="NOT FOUND"
                    fi
                    TOTAL_SIZE=$((SERVER_SIZE + CLIENT_SIZE))
                    TOTAL_HUMAN=$(bytes_to_human $TOTAL_SIZE)
                    DEALER_TIME="(skipped)"
                fi

                # ========== 按线程列表执行 Server/Client ==========
                for THREADS in "${THREADS_LIST[@]}"; do
                    THREAD_SUFFIX="_t${THREADS}"

                    echo ""
                    echo "    >>> Threads: $THREADS (t${THREADS})"

                    # 检查是否已存在结果
                    if [ -f "$RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}${THREAD_SUFFIX}_summary.txt" ]; then
                        echo "    [SKIP] Results for EPS=$eps, Image=$IMAGE_ID, Mode=$MODE_DISPLAY, Threads=$THREADS already exist"
                        echo "    Quick preview:"
                        tail -5 "$RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}${THREAD_SUFFIX}_summary.txt" | sed 's/^/    /'
                        echo ""
                        continue
                    fi

                    # 等待一下确保端口释放
                    echo "    [$(date +'%H:%M:%S')] Waiting for port to be available..."
                    sleep 3

                    # 启动 SERVER 和 CLIENT
                    echo "    [$(date +'%H:%M:%S')] Starting SERVER and CLIENT ($MODE_DISPLAY, Threads=$THREADS)..."

                    # 先启动 SERVER（在后台）
                    if [ "$THREADS" = "0" ]; then
                        ./build/benchmark-crown 0 $IP_ADDRESS $CUSTOM_ARGS &> $RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}${THREAD_SUFFIX}_server.txt &
                    else
                        OMP_NUM_THREADS=$THREADS ./build/benchmark-crown 0 $IP_ADDRESS $CUSTOM_ARGS &> $RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}${THREAD_SUFFIX}_server.txt &
                    fi
                    SERVER_PID=$!

                    # 等待 SERVER 准备好监听
                    sleep 2

                    # 启动 CLIENT
                    if [ "$THREADS" = "0" ]; then
                        ./build/benchmark-crown 1 $IP_ADDRESS $CUSTOM_ARGS &> $RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}${THREAD_SUFFIX}_client.txt
                    else
                        OMP_NUM_THREADS=$THREADS ./build/benchmark-crown 1 $IP_ADDRESS $CUSTOM_ARGS &> $RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}${THREAD_SUFFIX}_client.txt
                    fi
                    CLIENT_EXIT_CODE=$?

                    # 等待 SERVER 完成
                    wait $SERVER_PID 2>/dev/null || true

                    echo "    [$(date +'%H:%M:%S')] Computation completed! ($MODE_DISPLAY, Threads=$THREADS)"

                    # 检查是否成功
                    if [ $CLIENT_EXIT_CODE -ne 0 ]; then
                        echo "    WARNING: Client exited with code $CLIENT_EXIT_CODE"
                        echo "    Server output:"
                        cat $RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}${THREAD_SUFFIX}_server.txt | sed 's/^/    /'
                        echo ""
                        echo "    Client output:"
                        cat $RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}${THREAD_SUFFIX}_client.txt | sed 's/^/    /'
                    fi

                    # 创建摘要
                    {
                        echo "Results Summary"
                        echo "==============="
                        echo "Model: $model"
                        echo "Layers: $NUM_LAYERS, Hidden: $HIDDEN_DIM"
                        echo "Input: $INPUT_DIM, Output: $OUTPUT_DIM"
                        echo "EPS: $eps"
                        echo "Image ID: $IMAGE_ID"
                        echo "True Label: $TRUE_LABEL, Target Label: $TARGET_LABEL"
                        echo "Mode: $MODE_DISPLAY"
                        echo "Threads: $THREADS (0=system default)"
                        echo ""
                        echo "Preprocessing:"
                        echo "  Duration: $DEALER_TIME seconds"
                        echo "  server.dat: $SERVER_HUMAN"
                        echo "  client.dat: $CLIENT_HUMAN"
                        echo "  Total: $TOTAL_HUMAN"
                        echo ""
                        echo "Computation Results:"
                        # 提取并格式化结果，添加单位转换
                        while IFS= read -r line; do
                            if [[ "$line" =~ ^(End_to_end_time|crown_calculation|input):\ ([0-9]+)\ ms,\ ([0-9.]+)\ KB$ ]]; then
                                name="${BASH_REMATCH[1]}"
                                ms="${BASH_REMATCH[2]}"
                                kb="${BASH_REMATCH[3]}"
                                sec=$(awk "BEGIN {printf \"%.3f\", $ms/1000}")
                                mb=$(awk "BEGIN {printf \"%.3f\", $kb/1024}")
                                echo "$line  ($sec s, $mb MB)"
                            else
                                echo "$line"
                            fi
                        done < <(grep -E "(MPC LB:|MPC UB:|End_to_end_time|crown_calculation|input:)" $RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}${THREAD_SUFFIX}_client.txt 2>/dev/null) || echo "  (No results found - check for errors)"
                    } > $RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}${THREAD_SUFFIX}_summary.txt

                    echo ""
                    cat $RESULTS_BASE_DIR/$model/eps_$eps/image_${IMAGE_ID}${MODE_SUFFIX}${THREAD_SUFFIX}_summary.txt | sed 's/^/    /'
                    echo ""

                    # 每次测试后等待端口释放
                    sleep 3
                done
            done
        done
    done

    # 创建模型总结 (修改版：扫描磁盘上所有存在的 EPS 和 Image 结果，区分恶意和半诚实以及线程数)
    {
        echo "Model Summary: $model"
        echo "====================="
        echo "Layers: $NUM_LAYERS, Hidden: $HIDDEN_DIM"
        echo "Input: $INPUT_DIM, Output: $OUTPUT_DIM"
        echo "Threads List: ${THREADS_LIST[*]}"
        echo ""
        echo "Test samples in current config (id,true_class,target_class):"
        for sample in "${IMAGE_LIST[@]}"; do
            echo "  $sample"
        done
        echo ""
        echo "--- All Results Found in $RESULTS_BASE_DIR/$model ---"

        # 1. 查找所有 eps_ 开头的目录，并按版本号排序
        EXISTING_EPS_DIRS=$(find "$RESULTS_BASE_DIR/$model" -maxdepth 1 -type d -name "eps_*" | sort -V)

        if [ -z "$EXISTING_EPS_DIRS" ]; then
            echo "No results found on disk."
        else
            for eps_dir in $EXISTING_EPS_DIRS; do
                # 从文件夹路径中提取 eps 值
                eps_val=$(basename "$eps_dir" | sed 's/eps_//')

                echo "=== EPS = $eps_val ==="

                # 获取所有唯一的 image ID（支持新旧格式）
                ALL_IMAGE_IDS=$(find "$eps_dir" -maxdepth 1 -name "image_*_client.txt" -o -name "image_*_m_*_client.txt" -o -name "image_*_sh_*_client.txt" 2>/dev/null | \
                    sed -E 's/.*image_([0-9]+)_.*/\1/' | sort -n | uniq)

                if [ -z "$ALL_IMAGE_IDS" ]; then
                    echo "  (No client output files found in this folder)"
                else
                    for img_id in $ALL_IMAGE_IDS; do
                        # 尝试从配置中查找该 ID 对应的标签
                        config_str="${IMAGE_CONFIGS[$model]}"
                        matched_line=$(echo "$config_str" | grep -E "^\s*${img_id},")

                        if [ -n "$matched_line" ]; then
                            IFS=',' read -r _id _true _target <<< $(echo "$matched_line" | tr -d ' ')
                            echo "--- Image $_id (true=$_true, target=$_target) ---"
                        else
                            echo "--- Image $img_id (Labels not in current config) ---"
                        fi

                        # 查找该image的所有client文件并分类输出
                        # 恶意模式
                        MAL_FILES=$(find "$eps_dir" -maxdepth 1 -name "image_${img_id}_m_t*_client.txt" 2>/dev/null | sort -V)
                        if [ -n "$MAL_FILES" ]; then
                            echo "  [Malicious Mode]"
                            for mal_file in $MAL_FILES; do
                                # 提取线程数
                                thread_num=$(basename "$mal_file" | sed -E 's/.*_t([0-9]+)_client.txt/\1/')
                                echo "    Threads=$thread_num:"
                                while IFS= read -r line; do
                                    if [[ "$line" =~ ^(End_to_end_time|crown_calculation|input):\ ([0-9]+)\ ms,\ ([0-9.]+)\ KB$ ]]; then
                                        name="${BASH_REMATCH[1]}"
                                        ms="${BASH_REMATCH[2]}"
                                        kb="${BASH_REMATCH[3]}"
                                        sec=$(awk "BEGIN {printf \"%.3f\", $ms/1000}")
                                        mb=$(awk "BEGIN {printf \"%.3f\", $kb/1024}")
                                        echo "      $line  ($sec s, $mb MB)"
                                    else
                                        echo "      $line"
                                    fi
                                done < <(grep -E "(MPC LB:|MPC UB:|End_to_end_time|crown_calculation|input:)" "$mal_file" 2>/dev/null) || echo "      (No results content)"
                            done
                        fi

                        # 半诚实模式
                        SH_FILES=$(find "$eps_dir" -maxdepth 1 -name "image_${img_id}_sh_t*_client.txt" 2>/dev/null | sort -V)
                        if [ -n "$SH_FILES" ]; then
                            echo "  [Semi-Honest Mode]"
                            for sh_file in $SH_FILES; do
                                # 提取线程数
                                thread_num=$(basename "$sh_file" | sed -E 's/.*_t([0-9]+)_client.txt/\1/')
                                echo "    Threads=$thread_num:"
                                while IFS= read -r line; do
                                    if [[ "$line" =~ ^(End_to_end_time|crown_calculation|input):\ ([0-9]+)\ ms,\ ([0-9.]+)\ KB$ ]]; then
                                        name="${BASH_REMATCH[1]}"
                                        ms="${BASH_REMATCH[2]}"
                                        kb="${BASH_REMATCH[3]}"
                                        sec=$(awk "BEGIN {printf \"%.3f\", $ms/1000}")
                                        mb=$(awk "BEGIN {printf \"%.3f\", $kb/1024}")
                                        echo "      $line  ($sec s, $mb MB)"
                                    else
                                        echo "      $line"
                                    fi
                                done < <(grep -E "(MPC LB:|MPC UB:|End_to_end_time|crown_calculation|input:)" "$sh_file" 2>/dev/null) || echo "      (No results content)"
                            done
                        fi

                        # 兼容旧格式（无线程后缀）
                        OLD_MAL_FILE="$eps_dir/image_${img_id}_m_client.txt"
                        if [ -f "$OLD_MAL_FILE" ]; then
                            echo "  [Malicious Mode - Legacy (no thread info)]"
                            while IFS= read -r line; do
                                if [[ "$line" =~ ^(End_to_end_time|crown_calculation|input):\ ([0-9]+)\ ms,\ ([0-9.]+)\ KB$ ]]; then
                                    name="${BASH_REMATCH[1]}"
                                    ms="${BASH_REMATCH[2]}"
                                    kb="${BASH_REMATCH[3]}"
                                    sec=$(awk "BEGIN {printf \"%.3f\", $ms/1000}")
                                    mb=$(awk "BEGIN {printf \"%.3f\", $kb/1024}")
                                    echo "    $line  ($sec s, $mb MB)"
                                else
                                    echo "    $line"
                                fi
                            done < <(grep -E "(MPC LB:|MPC UB:|End_to_end_time|crown_calculation|input:)" "$OLD_MAL_FILE" 2>/dev/null) || echo "    (No results content)"
                        fi

                        OLD_SH_FILE="$eps_dir/image_${img_id}_sh_client.txt"
                        if [ -f "$OLD_SH_FILE" ]; then
                            echo "  [Semi-Honest Mode - Legacy (no thread info)]"
                            while IFS= read -r line; do
                                if [[ "$line" =~ ^(End_to_end_time|crown_calculation|input):\ ([0-9]+)\ ms,\ ([0-9.]+)\ KB$ ]]; then
                                    name="${BASH_REMATCH[1]}"
                                    ms="${BASH_REMATCH[2]}"
                                    kb="${BASH_REMATCH[3]}"
                                    sec=$(awk "BEGIN {printf \"%.3f\", $ms/1000}")
                                    mb=$(awk "BEGIN {printf \"%.3f\", $kb/1024}")
                                    echo "    $line  ($sec s, $mb MB)"
                                else
                                    echo "    $line"
                                fi
                            done < <(grep -E "(MPC LB:|MPC UB:|End_to_end_time|crown_calculation|input:)" "$OLD_SH_FILE" 2>/dev/null) || echo "    (No results content)"
                        fi

                        # 兼容更旧格式（无 _m 或 _sh 后缀）
                        OLD_FILE="$eps_dir/image_${img_id}_client.txt"
                        if [ -f "$OLD_FILE" ]; then
                            echo "  [Legacy Format - Mode & Thread Unknown]"
                            while IFS= read -r line; do
                                if [[ "$line" =~ ^(End_to_end_time|crown_calculation|input):\ ([0-9]+)\ ms,\ ([0-9.]+)\ KB$ ]]; then
                                    name="${BASH_REMATCH[1]}"
                                    ms="${BASH_REMATCH[2]}"
                                    kb="${BASH_REMATCH[3]}"
                                    sec=$(awk "BEGIN {printf \"%.3f\", $ms/1000}")
                                    mb=$(awk "BEGIN {printf \"%.3f\", $kb/1024}")
                                    echo "    $line  ($sec s, $mb MB)"
                                else
                                    echo "    $line"
                                fi
                            done < <(grep -E "(MPC LB:|MPC UB:|End_to_end_time|crown_calculation|input:)" "$OLD_FILE" 2>/dev/null) || echo "    (No results content)"
                        fi

                        echo ""
                    done
                fi
            done
        fi
    } > $RESULTS_BASE_DIR/$model/model_summary.txt
done

echo ""
echo "=========================================================="
echo "All tests completed!"
echo "Network condition: $NETWORK_CONDITION"
echo "Results saved in: $RESULTS_BASE_DIR/"
echo "=========================================================="