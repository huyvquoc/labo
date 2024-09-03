#!/bin/bash
# $1: number of shots
# $2: dataset (flower/food101)

export PYTHONPATH="${PYTHONPATH}:/data/hqvo3/codes/CONCH"
shots='all'
dataset="2_PanCancerTCGA"

CUDA_VISIBLE_DEVICES=6 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac_with_wb_nosoft_onlyconcepts.py --work-dir exp/concept_trans/${dataset}/${dataset}_${shots}shot_fac_with_wb_macro --func concept_trans_main