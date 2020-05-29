#!/bin/bash

#SBTACH --time=23:00:00
#SBATCH --ntasks=1 
#SBATCH --account=ctb-liyue
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output slurm_output/slurn-%j.out
#SBATCH -c 10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yueli.cs@gmail.com

python main.py \
    --mode train \
    --dataset Aylien \
    --data_path /home/liyue/projects/ctb-liyue/liyue/Projects/covid19_media/data/Aylien \
    --emb_path /home/liyue/projects/ctb-liyue/data/covid19_news/trained_word_emb_aylien.txt \
    --save_path /home/liyue/projects/ctb-liyue/liyue/Projects/covid19_media/results/dmetm \
    --lr 5e-4 \
    --clip 2.0 \
    --epochs 100 \
    --num_topics 50 \
    --batch_size 24 \
    --min_df 100 \
    --train_source_embeddings 0 \
    --eta_nlayers 3 \
    --eta_hidden_size 512 \
    --t_hidden_size 512 \
    --eval_batch_size 100

