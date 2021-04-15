#!/bin/bash
n_obs=5
n_pred=10
#---------------data preprocessing parameters
#echo "starting data preprocessing........................."
#train_chunk_stride=8
#val_chunk_stride=12
#cd trajnetdataset
#python -m trajnetdataset.convert --n_obs $n_obs --n_pred $n_pred --train_chunk_stride $train_chunk_stride --val_chunk_stride $val_chunk_stride
#cd ..
#-----------------------------train & eval parameters
echo "starting training........................."
cd trajnetbaselines
n_epochs=14
init_lr=0.001
scene_mode='RRB_M'  #EDN,EDN_M,RRB,RRB_M
disable_cuda=0
train_input_files='../trajnetdataset/output_interaction_sceneGeneralization/train/' #'../trajnetdataset/output_interaction_sceneOverfitting/train/'
val_input_files='../trajnetdataset/output_interaction_sceneGeneralization/val_for_monitoring_training/'  #'../trajnetdataset/output_interaction_sceneOverfitting/val_for_monitoring_training/'
CUDA_VISIBLE_DEVICES=0 python -m trajnetbaselines.trainer --disable-cuda $disable_cuda --epochs $n_epochs --n_obs=$n_obs --n_pred=$n_pred --scene-mode $scene_mode --lr $init_lr --train-input-files $train_input_files --val-input-files $val_input_files
cd ..

