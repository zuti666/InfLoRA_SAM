#!/bin/bash
set -e  # 一旦出错就中止执行

export CUDA_VISIBLE_DEVICES=3


python main.py \
    --device 0 \
    --config configs/cifar100_inflora.json

python main.py \
    --device 0 \
    --rho 0.05 \
    --config configs/cifar100_inflora-sam.json


