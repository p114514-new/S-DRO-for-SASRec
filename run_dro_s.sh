#!/usr/bin/env bash

python run_recbole_dro_s.py --model=DRO_S --dataset=ml-100k --train_neg_sample_args=None --gpu_id='0'
python run_recbole_dro_s.py --model=DRO_S --dataset=retailrocket-view --train_neg_sample_args=None --gpu_id='0'
python run_recbole_dro_s.py --model=DRO_S --dataset=sports --train_neg_sample_args=None --gpu_id='0'
python run_recbole_dro_s.py --model=DRO_S --dataset=beauty --train_neg_sample_args=None --gpu_id='0'
