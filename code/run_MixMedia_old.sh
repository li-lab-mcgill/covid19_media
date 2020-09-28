#!/bin/bash

K=$1
predict_cnpi=$2
seed=$3
cuda=$4

cd MixMedia_old

dataset=WHO_910
# dataset=CNET
q_theta_arc=lstm
# dataset=GPHIN_all
# datadir=/home/mcb/users/zwen8/data/covid/test_who_all/
datadir=/home/mcb/users/zwen8/data/covid/${dataset}_cnpi_${seed}/
# datadir=/home/mcb/users/zwen8/data/covid/data_June16/WHO_all
outdir=/home/mcb/users/zwen8/data/covid/results/cnpi/${dataset}/old/${q_theta_arc}/seed_${seed}
wemb=/home/mcb/users/zwen8/data/covid/csv_data_files/skipgram_emb_300d.txt

if [ ! -d $outdir ]; then
	mkdir $outdir -p
fi

CUDA_VISIBLE_DEVICES=${cuda} python main.py \
    --mode train \
    --dataset ${dataset} \
    --data_path $datadir \
    --batch_size 128 \
    --emb_path $wemb \
    --save_path $outdir \
    --lr 1e-3 \
    --epochs 400 \
    --num_topics $K \
    --min_df 10 \
    --train_embeddings 1 \
    --eval_batch_size 128 \
    --time_prior 1 \
    --source_prior 1 \
    --predict_cnpi $predict_cnpi \
    --multiclass_labels 1 \
    --one_hot_qtheta_emb 1 \
    --q_theta_arc ${q_theta_arc} \
    --cnpi_layers 3 \
    --cnpi_hidden_size 128 \
    --predict_labels 0 \
    # --q_theta_hidden_size 256 \
    # --q_theta_heads 3 \
    # --q_theta_layers 3 \
    # --load_from /home/zwen8/projects/ctb-liyue/zwen8/covid_run/results/June16/MixMedia/${dataset}/mixmedia_${dataset}_K_${K}_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_3_minDF_10_trainEmbeddings_${train_word_emb}_predictLabels_${predict_labels}_useTime_1_useSource_1
# --load_from /home/zwen8/projects/ctb-liyue/zwen8/covid_run/results/MixMedia/WHO_all/mixmedia_WHO_all_K_${K}_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_3_minDF_10_trainEmbeddings_${train_word_emb}_predictLabels_0 \
    # --load_from /home/zwen8/projects/ctb-liyue/zwen8/covid_run/results/MixMedia/GPHIN_all/mixmedia_GPHIN_all_K_${K}_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_3_minDF_10_trainEmbeddings_${train_word_emb}_predictLabels_0 \
    # --load_from ${outdir}/detm_${dataset}_K_${K}_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_3_minDF_10_trainWordEmbeddings_${train_word_emb}_epochs_1000 \
    # --load_from /home/liyue/projects/ctb-liyue/liyue/Projects/covid19_media/results/method_comparison/detm_${dataset}_K_${K}_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_3_minDF_10_trainWordEmbeddings_${train_word_emb} \
