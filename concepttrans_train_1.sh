#!/bin/bash
# $1: number of shots
# $2: dataset (flower/food101)

export PYTHONPATH="${PYTHONPATH}:/data/hqvo3/codes/CONCH"
shots='all'
dataset="GlomCrawl"

# CUDA_VISIBLE_DEVICES=7 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac.py --work-dir exp/asso_opt/${dataset}/${dataset}_${shots}shot_fac --func asso_opt_main


CUDA_VISIBLE_DEVICES=3 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac_with_wb.py --work-dir exp/concept_trans/${dataset}/${dataset}_${shots}shot_fac_wo_wb_lsmoothing2_openclip_bs4096_5softs_wb --func concept_trans_main
# CUDA_VISIBLE_DEVICES=7 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac_with_wb.py --work-dir exp/concept_trans/${dataset}/${dataset}_${shots}shot_fac_wo_wb_lsmoothing2_evaclip_bs4096_5softs_wb --func concept_trans_main

# CUDA_VISIBLE_DEVICES=5 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac_with_wb.py --work-dir exp/concept_trans/${dataset}/${dataset}_${shots}shot_fac_with_wb_lsmoothing1 --func concept_trans_main

# CUDA_LAUNCH_BLOCKING=1 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac_wo_wb.py --work-dir exp/concept_trans/${dataset}/${dataset}_${shots}shot_fac_wo_wb_lsmoothing2_conch_bs4096_5softs_use_attn_branch_all_concepts --func concept_trans_main



