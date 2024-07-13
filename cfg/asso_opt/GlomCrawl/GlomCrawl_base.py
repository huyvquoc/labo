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
bs = 64
on_gpu = True
use_txt_norm = True
use_img_norm = True

# concept select
num_concept = num_cls * 15

# weight matrix fitting
lr = 5e-6
max_epochs = 100

# weight matrix
use_rand_init = False
init_val = 1.
asso_act = 'softmax'
use_l1_loss = True
use_div_loss = True
lambda_l1 = 0
lambda_div = 0
# lambda_l1 = 0.0001
# lambda_div = 0.0005

# CLIP Backbone
# clip_model = 'ViT-L/14'
# clip_model_name = 'clip'
clip_model = 'conch_ViT-B-16'
clip_model_name = 'conch'
top_k = 5
balanced_weights = [0.6863153028201572, 2.8276190476190477, 0.9301378446115288, 0.6668912848158132, 2.0791316526610646, 3.7205513784461153, 0.4502578101304216, 5.049319727891157, \
                    0.8948161543098252, 0.3086920357662716, 1.0098639455782312, 1.5367494824016563, 2.5246598639455784, 0.6200918964076859, 5.049319727891157, 0.3086920357662716, \
                    0.728767795778105, 1.3594322344322345, 2.0197278911564625, 2.5246598639455784, 8.836309523809524, 0.5890873015873016, 1.4426627793974733, 4.158263305322129, \
                    3.5345238095238094, 1.4426627793974733, 0.3927248677248677, 0.21292312105565117, 0.9301378446115288, 0.728767795778105, 1.1781746031746032, 1.5367494824016563, \
                    3.7205513784461153, 0.9683626875407697, 1.472718253968254, 10.098639455782314, 2.8276190476190477, 0.9683626875407697, 5.049319727891157, 1.1588602654176425, \
                    8.836309523809524, 2.209077380952381]
main_loss = 'TwoWayLoss'
# main_loss = 'AsymmetricLossOptimized'
# main_loss = 'FocalLoss'
# main_loss = 'APLLoss'
# main_loss = 'ANLSLoss'