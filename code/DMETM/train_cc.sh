#!/bin/bash

#SBTACH --time=1:0:0 
#SBATCH --ntasks=1 
#SBATCH --account=ctb-liyue
#SBATCH --gres=gpu:1
#SBATCH --mem=125G

CUDA_VISIBLE_DEVICES=1 python main.py \
    --mode train \
    --dataset Aylien \
    --data_path /home/liyue/projects/ctb-liyue/liyue/Projects/covid19_media/data/Aylien \
    --emb_path /home/liyue/projects/ctb-liyue/data/covid19_news/trained_word_emb_aylien.txt \
    --save_path /home/liyue/projects/ctb-liyue/liyue/Projects/covid19_media/results/dmetm \
    --lr 8e-3 \
    --clip 2.0 \
    --epochs 20 \
    --num_topics 40 \
    --batch_size 64 \
    --min_df 100 \
    --train_source_embeddings 0 \
    --eta_nlayers 3 \
    --eta_hidden_size 512 \
    --t_hidden_size 512 \
    --eval_batch_size 100


# python main.py --min_df 100 --dataset Aylien --data_path /home/liyue/projects/ctb-liyue/liyue/Projects/covid19_media/data/Aylien --num_topics 50 --t_hidden_size 64 --eta_hidden_size 64 --batch_size 100 --epochs 100 --emb_path /home/liyue/projects/ctb-liyue/data/covid19_news/trained_word_emb_aylien.txt --eval_batch_size 100 --save_path /home/liyue/projects/ctb-liyue/liyue/Projects/covid19_media/results/dmet --clip 2.0

