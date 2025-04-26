#!/bin/bash

# 设置使用的 GPU
export CUDA_VISIBLE_DEVICES=3  # 可改为 0、1 等 GPU 编号

# 设置日志路径
LOG_DIR="logs_probe/cifar10"  # 根据你的 probe 数据集修改此目录
mkdir -p "$LOG_DIR"

# 当前时间戳用于唯一标识日志文件
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/probe_$TIMESTAMP.log"

# 启动 linear-probe 训练脚本
python linear_prob/linear_probe.py \
    --mode multi \
    --config linear_prob/config_multi_linear_probe.json \
    > "$LOG_FILE" 2>&1 &

echo "Linear probe script started. Logs are being written to: $LOG_FILE"
