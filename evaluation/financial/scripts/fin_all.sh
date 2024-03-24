#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=$2 python benchmarks/fingpt_bench.py \
    --model $1 \
    --batch_size 8 \
    --max_length 512 
