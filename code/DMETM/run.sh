#!/bin/bash

python main.py \
    --mode train \
    --dataset "GPHIN" \
    --data_path data/GPHIN/min_df_10/ \
    --num_topics 10 \
    --train_embeddings 1 \
    --epochs 10