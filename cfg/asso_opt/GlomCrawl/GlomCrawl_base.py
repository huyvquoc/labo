_base_ = '../base.py'
# dataset
proj_name = "GlomCrawl"
concept_root = 'datasets/GlomCrawl/concepts/'
img_split_path = 'datasets/GlomCrawl/splits'
img_path = '/data/hqvo3/Datasets/vlm/ml-scripts/datasets/glom_refine_labo'
concept_type = "all"
img_ext = '.png'
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 42

## data loader
bs = 128
on_gpu = True

# concept select
num_concept = num_cls * 15

# weight matrix fitting
lr = 5e-6
max_epochs = 10000

# weight matrix
use_rand_init = False
init_val = 1.
asso_act = 'softmax'
use_l1_loss = True
use_div_loss = True
lambda_l1 = 0.01
lambda_div = 0.005

# CLIP Backbone
# clip_model = 'ViT-L/14'
# clip_model_name = 'clip'
clip_model = 'conch_ViT-B-16'
clip_model_name = 'conch'