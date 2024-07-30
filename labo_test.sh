# $1: cfg path
# $2: ckpt path

export PYTHONPATH="${PYTHONPATH}:/data/hqvo3/codes/CONCH"

# cfg_path="/data/hqvo3/codes/LaBo/exp/asso_opt/GlomCrawl/GlomCrawl_allshot_fac/GlomCrawl_allshot_fac.py"
ckpt_path="/data/hqvo3/codes/LaBo/exp/asso_opt/GlomCrawl/GlomCrawl_allshot_fac/epoch=749-step=11250-val_acc=0.3636.ckpt"
shots='all'
dataset="GlomCrawl"

CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac.py --func concept_trans_main --test --cfg-options bs=1024 ckpt_path=${ckpt_path}
#  python main.py --cfg cfg/asso_opt/${dataset}/${dataset}_${shots}shot_fac.py --work-dir exp/asso_opt/${dataset}/${dataset}_${shots}shot_fac --func concept_trans_main