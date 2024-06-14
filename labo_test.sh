# $1: cfg path
# $2: ckpt path
cfg_path="/data/hqvo3/codes/LaBo/exp/asso_opt/GlomCrawl/GlomCrawl_allshot_fac/GlomCrawl_allshot_fac.py"
ckpt_path="/data/hqvo3/codes/LaBo/exp/asso_opt/GlomCrawl/GlomCrawl_allshot_fac/epoch=1029-step=3090-val_acc=0.3162.ckpt"

python main.py --cfg ${cfg_path} --func asso_opt_main --test --cfg-options bs=1024 ckpt_path=${ckpt_path} ${@:3}