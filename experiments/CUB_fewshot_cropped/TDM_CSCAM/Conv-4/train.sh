#!/bin/bash
# TDM+CSCAM
python train.py --TDM --noise --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 2 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 20 --train_shot 5 --train_transform_type 0 --test_shot 1 5 --pre --gpu 5
python train.py --TDM --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 2 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 10 --train_shot 5 --train_transform_type 0 --test_shot 1 5 --pre --gpu 5


