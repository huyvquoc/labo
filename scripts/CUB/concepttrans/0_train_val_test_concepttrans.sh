#!/bin/bash
# $1: number of shots
# $2: dataset (flower/food101)

export PYTHONPATH="${PYTHONPATH}:/data/hqvo3/codes/CONCH"
shots='all'
dataset="CUB"

CUDA_VISIBLE_DEVICES=5 python main.py --cfg cfg/concept_trans/${dataset}/CUB_${shots}shot_fac.py --work-dir exp/concept_trans/${dataset}/${dataset}_${shots}shot_fac --func concept_trans_main