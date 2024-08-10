
#!/bin/bash
# $1: number of shots
# $2: dataset (flower/food101)

export PYTHONPATH="${PYTHONPATH}:/data/hqvo3/codes/CONCH"
shots='all'
dataset="GlomCrawl"

CUDA_VISIBLE_DEVICES=7 python main.py --cfg cfg/asso_opt/${dataset}/labo_${shots}shot_xent.py --work-dir exp/asso_opt_re/${dataset}/${dataset}_${shots}shot_fac --func asso_opt_main
# CUDA_VISIBLE_DEVICES=7 python main.py --cfg cfg/asso_opt/${dataset}/labo_${shots}shot_bce.py --work-dir exp/asso_opt_re/${dataset}/${dataset}_${shots}shot_fac --func asso_opt_main