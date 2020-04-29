#!/bin/bash

python main.py \
    --mode train \
    --dataset "GPHIN" \
    --data_path data/GPHIN \
    --num_topics 10 \
    --train_embeddings 1 \
    --epochs 10