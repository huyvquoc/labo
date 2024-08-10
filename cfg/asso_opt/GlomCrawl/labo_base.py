_base_ = '../base.py'
# dataset
proj_name = "GlomCrawl"
concept_root = 'datasets/GlomCrawl/concepts/' # TODO: change datasets - relative path - manually generate `class2concepts.json`
img_split_path = 'datasets/GlomCrawl/splits' # TODO: change datasets - relative path - manually generate pickle files (.p) of train/val/test
img_path = '/data/hqvo3/Datasets/vlm/ml-scripts/datasets/glom_refine_labo' # TODO: change datasets - absolute path - the folder contains all images of the dataset (no subfolders of train/val/test)
concept_type = "all"
img_ext = '.png'
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 42

## data loader
bs = 2048
on_gpu = True
use_txt_norm = True
use_img_norm = True

# concept select
num_concept = num_cls * 15 # TODO: change datasets

# weight matrix fitting
lr = 1e-2
max_epochs = 1000

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
# clip_model = 'ViT-L-14'
clip_model_name = 'conch'
# main_loss = 'TwoWayLoss'
# main_loss = 'AsymmetricLoss'
# main_loss = 'AsymmetricLossOptimized'
# main_loss = 'FocalLoss'
# main_loss = 'APLLoss'
main_loss = 'ANLSLoss'

groupname2clsname = {"Glomerular Basement Membrane (GBM), Bowman's Capsule, Epithelial Cells": ['glomerular basement membrane duplication splitting',
                                                                                                'glomerular basement membrane thickening',
                                                                                                'hyalin drop diabetic glomerulus',
                                                                                                'membranous nephropathy holes',
                                                                                                'membranous nephropathy spikes',
                                                                                                'podocyte hyperplasia',
                                                                                                'podocyte hypertrophy',
                                                                                                'podocyte reabsorption droplet',
                                                                                                'spikes glomerular amyoidosis',
                                                                                                'visceral epithelial cell capping', \
                                                                                                'wire loop glomerulus',],\
                     "Mesangial and Endocapillary Lesions": ['endocapillary hypercellularity',
                                                             'exudative glomerulus',
                                                             'glomerul lipoprotein thrombus',
                                                             'glomerular congestion',
                                                             'glomerular hyaline pseudothrombus',
                                                             'glomerular thrombus',
                                                             'glomerulonephritis hyaline thrombus',
                                                             'glomerulus endotheliosis',
                                                             'mesangial hypercellularity',
                                                             'mesangial matrix expansion',
                                                             'mesangiolysis'], \
                     "Miscellaneous Lesions and Key Normal Structures": ['glomerulomegaly',
                                                                         'immature glomerulus',
                                                                         'parietal cell glomerulus',
                                                                         'podocyte',
                                                                         'visceral epithelial cell'], \
                     "Necrotizing and Crescentic Lesions": ['fibrinoid necrosis glomerulus',
                                                            'fibrocellular crescent', # subgroup: crescents
                                                            'fibrous crescent', # subgroup: crescents
                                                            'karyorrhexis glomerulonephritis',
                                                            'pseudocrescent'], \
                     "Tuft Sclerosis and Collapse": ['glomerular adhesion',
                                                     'glomerular collapse', # subgroup: capillary collapse
                                                     'glomerular foam cell',
                                                     'glomerular hyalinosis',
                                                     'glomerular obsolesce',
                                                     'glomerular segmental sclerosis', # subgroup: segmental sclerosis
                                                     'ischemic glomerular collapse', # subgroup: capillary collapse
                                                     'nodular sclerosis glomerulus', # subgroup: sclerosis
                                                     'periglomerular fibrosis',
                                                     'solidified glomerular sclerosis global'] # subgroup: sclerosis
}