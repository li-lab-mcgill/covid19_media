#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode train \
    --dataset "Aylien" \
    --data_path data/Aylien/ \
    --emb_path data/trained_word_emb_aylien.txt \
    --save_path ./results \
    --lr 1e-3 \
    --clip 2 \
    --enc_drop 0.1 \
    --eta_dropout 0.1 \
    --epochs 40 \
    --num_topics 50 \
    --batch_size 32 \
    --min_df 100 \
    --eta_nlayers 3 \
    --eta_hidden_size 512 \
    --t_hidden_size 512 \
    --eval_batch_size 100 \
