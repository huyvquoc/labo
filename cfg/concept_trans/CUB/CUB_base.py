_base_ = '../base.py'
# dataset 
proj_name = "CUB"
img_split_path = 'datasets/CUB/splits'
img_path = '/data/hqvo3/data/cub2011/CUB_200_2011/images'

concept_type = "all"

img_ext = ''
num_cls = 200

## data loader
bs = 32
on_gpu = True

# concept select
num_concept = num_cls * 50

# weight matrix fitting
lr = 1e-5
max_epochs = 5000

# weight matrix
use_rand_init = False
init_val = 1.
asso_act = 'softmax'
use_l1_loss = False
use_div_loss = False
lambda_l1 = 0.01
lambda_div = 0.005

# CLIP Backbone
# clip_model = 'conch_ViT-B-16'
# clip_model_weight = None
# clip_model = 'ViT-L-14'
# clip_model_weight = 'laion2b_s32b_b82k'
clip_model = 'EVA02-L-14'
clip_model_weight = 'merged2b_s4b_b131k' # only for OpenCLIP library, set to None if do not use OpenCLIP

# clip_model_name = 'clip'
clip_model_name = 'conch'

# main_loss = 'TwoWayLoss'
# main_loss = 'AsymmetricLoss'
# main_loss = 'AsymmetricLossOptimized'
# main_loss = 'FocalLoss'
# main_loss = 'APLLoss'
main_loss = 'ANLSLoss'

groupname2clsname = {}