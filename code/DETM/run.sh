#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode train \
    --dataset "Aylien" \
    --data_path data/Aylien/ \
    --emb_path data/trained_word_emb_aylien.txt \
    --save_path ./results \
    --lr 5e-4 \
    --clip 2 \
    --epochs 100 \
    --num_topics 40 \
    --batch_size 64 \
    --min_df 100 \
    --train_source_embeddings 0 \
    --eta_nlayers 3 \
    --eta_hidden_size 512 \
    --t_hidden_size 512 \
    --eval_batch_size 100 \