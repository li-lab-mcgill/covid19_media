#!/bin/bash

CUDA_VISIBLE_DEVICES=9 python main.py \
    --mode train \
    --dataset "Aylien" \
    --data_path data/Aylien/ \
    --emb_path data/trained_word_emb_aylien.txt \
    --save_path ./results \
    --epochs 100 \
    --num_topics 10 \
    --batch_size 16 \
    --min_df 100 \
    --train_source_embeddings 0 \
    --eta_nlayers 3 \
    --eta_hidden_size 64 \
    --t_hidden_size 64 \
    --eval_batch_size 100 \