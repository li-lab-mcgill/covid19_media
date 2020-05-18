#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode train \
    --dataset "Aylien" \
    --data_path data/Aylien/ \
    --emb_path skipgram/trained_word_emb_aylien.txt \
    --save_path ./results \
    --epochs 100 \
    --num_topics 50 \
    --batch_size 8 \
    --min_df 100 \
    --train_source_embeddings 0 \
    --eta_nlayers 2 \
    --eta_hidden_size 5 \
    --t_hidden_size 5 \
    --eval_batch_size 100 \