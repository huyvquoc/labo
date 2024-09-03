_base_ = '../base.py'
# dataset
proj_name = "2_PanCancerTCGA"
concept_root = 'datasets/2_PanCancerTCGA/concepts_bl4/' # TODO: change datasets - relative path - manually generate `class2concepts.json`
img_split_path = 'datasets/2_PanCancerTCGA/splits' # TODO: change datasets - relative path - manually generate pickle files (.p) of train/val/test
img_path = '/data/hqvo3/Datasets/histo_cls/2_PanCancerTCGA' # TODO: change datasets - absolute path - the folder contains all images of the dataset (no subfolders of train/val/test)
concept_type = "all"
img_ext = '.jpg'
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 32

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

groupname2clsname = {'Adrenocortical carcinoma': ['Adrenocortical carcinoma'],
                     'Bladder Urothelial Carcinoma': ['Bladder Urothelial Carcinoma'],
                     'Brain Lower Grade Glioma': ['Brain Lower Grade Glioma'],
                     'Breast invasive carcinoma': ['Breast invasive carcinoma'],
                     'Cervical squamous cell carcinoma and endocervical adenocarcinoma': ['Cervical squamous cell carcinoma and endocervical adenocarcinoma'],
                     'Cholangiocarcinoma': ['Cholangiocarcinoma'],
                     'Colon adenocarcinoma': ['Colon adenocarcinoma'],
                     'Esophageal carcinoma': ['Esophageal carcinoma'],
                     'Glioblastoma multiforme': ['Glioblastoma multiforme'],
                     'Head and Neck squamous cell carcinoma': ['Head and Neck squamous cell carcinoma'],
                     'Kidney Chromophobe': ['Kidney Chromophobe'],
                     'Kidney renal clear cell carcinoma': ['Kidney renal clear cell carcinoma'],
                     'Kidney renal papillary cell carcinoma': ['Kidney renal papillary cell carcinoma'],
                     'Liver hepatocellular carcinoma': ['Liver hepatocellular carcinoma'],
                     'Lung adenocarcinoma': ['Lung adenocarcinoma'],
                     'Lung squamous cell carcinoma': ['Lung squamous cell carcinoma'],
                     'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma': ['Lymphoid Neoplasm Diffuse Large B-cell Lymphoma'],
                     'Mesothelioma': ['Mesothelioma'],
                     'Ovarian serous cystadenocarcinoma': ['Ovarian serous cystadenocarcinoma'],
                     'Pancreatic adenocarcinoma': ['Pancreatic adenocarcinoma'],
                     'Pheochromocytoma and Paraganglioma': ['Pheochromocytoma and Paraganglioma'],
                     'Prostate adenocarcinoma': ['Prostate adenocarcinoma'],
                     'Rectum adenocarcinoma': ['Rectum adenocarcinoma'],
                     'Sarcoma': ['Sarcoma'],
                     'Skin Cutaneous Melanoma': ['Skin Cutaneous Melanoma'],
                     'Stomach adenocarcinoma': ['Stomach adenocarcinoma'],
                     'Testicular Germ Cell Tumors': ['Testicular Germ Cell Tumors'],
                     'Thymoma': ['Thymoma'],
                     'Thyroid carcinoma': ['Thyroid carcinoma'],
                     'Uterine Carcinosarcoma': ['Uterine Carcinosarcoma'],
                     'Uterine Corpus Endometrial Carcinoma': ['Uterine Corpus Endometrial Carcinoma'],
                     'Uveal Melanoma': ['Uveal Melanoma']}