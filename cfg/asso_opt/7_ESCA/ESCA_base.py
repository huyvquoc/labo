_base_ = '../base.py'
# dataset
proj_name = "7_ESCA"
concept_root = 'datasets/7_ESCA/concepts/' # TODO: change datasets - relative path - manually generate `class2concepts.json`
img_split_path = 'datasets/7_ESCA/splits' # TODO: change datasets - relative path - manually generate pickle files (.p) of train/val/test
img_path = '/data/hqvo3/Datasets/histo_cls/7_ESCA' # TODO: change datasets - absolute path - the folder contains all images of the dataset (no subfolders of train/val/test)
concept_type = "all"
img_ext = '.jpg'
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 11

## data loader
bs = 2048
on_gpu = True
use_txt_norm = True
use_img_norm = True

# concept select
num_concept = num_cls * 15 # TODO: change datasets

# weight matrix fitting
lr = 1e-2
max_epochs = 200

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
clip_model = 'conch_ViT-B-16'
clip_model_weight = None

# clip_model = 'ViT-L-14'
# clip_model_weight = 'laion2b_s32b_b82k'
# clip_model = 'EVA02-L-14'
# clip_model_weight = 'merged2b_s4b_b131k' # only for OpenCLIP library, set to None if do not use OpenCLIP

# clip_model_name = 'clip'
clip_model_name = 'conch'
 
# main_loss = 'TwoWayLoss'
# main_loss = 'AsymmetricLoss'
# main_loss = 'AsymmetricLossOptimized'
# main_loss = 'FocalLoss'
# main_loss = 'APLLoss'
main_loss = 'ANLSLoss'

groupname2clsname = {'adventitia': ['adventitia'],
                     'lamina propria mucosae': ['lamina propria mucosae'],
                     'muscularis mucosae': ['muscularis mucosae'],
                     'muscularis propria': ['muscularis propria'],
                     'regression tissue': ['regression tissue'],
                     'mucosa gastric': ['mucosa gastric'],
                     'mucosa oesophagus': ['mucosa oesophagus'],
                     'submucosa': ['submucosa'],
                     'submucosal glands': ['submucosal glands'],
                     'tumour': ['tumour'],
                     'ulceration': ['ulceration']}