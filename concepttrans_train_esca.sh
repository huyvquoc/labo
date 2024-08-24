#!/bin/bash
# $1: number of shots
# $2: dataset (flower/food101)

export PYTHONPATH="${PYTHONPATH}:/data/hqvo3/codes/CONCH"
shots='all'
dataset="7_ESCA"

CUDA_VISIBLE_DEVICES=5 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac_with_wb.py --work-dir exp/concept_trans/${dataset}/${dataset}_${shots}shot_fac_with_wb_macro --func concept_trans_main
# CUDA_VISIBLE_DEVICES=5 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac_wo_wb.py --work-dir exp/concept_trans/${dataset}/${dataset}_${shots}shot_fac_wo_wb_macro --func concept_trans_main
# CUDA_LAUNCH_BLOCKING=1 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac_wo_wb.py --work-dir exp/concept_trans/${dataset}/${dataset}_${shots}shot_fac_wo_wb_macro --func concept_trans_main
