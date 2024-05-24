# $1: cfg path
# $2: ckpt path
cfg_path="/data/hqvo3/codes/LaBo/exp/asso_opt/GlomCrawl/GlomCrawl_1shot_fac/GlomCrawl_1shot_fac.py"
ckpt_path="/data/hqvo3/codes/LaBo/exp/asso_opt/GlomCrawl/GlomCrawl_1shot_fac/epoch=1689-step=5070-val_acc=0.0384.ckpt"

python main.py --cfg ${cfg_path} --func asso_opt_main --test --cfg-options bs=512 ckpt_path=${ckpt_path} ${@:3}