#!/bin/bash
# $1: number of shots
# $2: dataset (flower/food101)

export PYTHONPATH="${PYTHONPATH}:/data/hqvo3/codes/CONCH"
shots='all'
dataset="GlomCrawl"

CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac_unfreeze_wo_wb.py --work-dir exp/concept_trans/${dataset}/${dataset}_${shots}shot_fac_unfreeze_wo_wb_lsmoothing1_version3 --func concept_trans_main