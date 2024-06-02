#!/bin/bash
# $1: number of shots
# $2: dataset (flower/food101)
shots=4
dataset="GlomCrawl"

CUDA_VISIBLE_DEVICES=7 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac.py --work-dir exp/asso_opt/${dataset}/${dataset}_${shots}shot_fac --func asso_opt_main