#!/bin/bash
python main.py --dataset aylien \
        --data_path ../../data/covid19_news/Aylien_DMETM/min_df_100/ \
        --save_path ../../data/covid19_news/Aylien_DMETM/Results_METM/ \
        --source_embedding_file ../../data/covid19_news/Aylien_DMETM/min_df_100/sources_matrix.npy \
        --sources_map_file ../../data/covid19_news/Aylien_DMETM/min_df_100/sources_map.pkl \
        --emb_path ../../data/covid19_news/trained_word_emb_aylien.txt \
        --t_hidden_size 64 \
        --num_topics 50 \
        --mode train \
        --epochs 200 \
        --batch_size 128

