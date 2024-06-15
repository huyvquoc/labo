# $1: cfg path
# $2: ckpt path
cfg_path="/data/hqvo3/codes/LaBo/exp/asso_opt/GlomCrawl/GlomCrawl_allshot_fac/GlomCrawl_allshot_fac.py"
ckpt_path="/data/hqvo3/codes/LaBo/exp/asso_opt/GlomCrawl/GlomCrawl_allshot_fac/epoch=6709-step=20130-val_acc=0.3172.ckpt"

python main.py --cfg ${cfg_path} --func asso_opt_main --test --cfg-options bs=1024 ckpt_path=${ckpt_path} ${@:3}