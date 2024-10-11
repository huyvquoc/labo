_base_ = 'CUB_base.py'
n_shots = "all"
data_root = 'exp/concept_trans/CUB/CUB_allshot_fac'
best_val_metric_name = 'val_main_cls_acc_multiclass_top1_macro' # for plotting only when model checkpoint
do_val_plotting=True
lr = 1e-3
bs = 256

concept_type = "all_submodular"
# concept_select_fn = "submodular"
concept_select_fn = "clip"
use_mi = False
group_select = True
submodular_weights = [1e7, 1]
# submodular_weights = [1e9, 0.7]

# ----- only for ConceptTransformer -----
weight_decay = 1e-12
disable_lr_scheduler = False
warmup = 500
group_lambda = 3.0
qk_group_lambda = 3.0
expl_lambda = 2.0
attention_sparsity = 0
ctc_model = 'cub_cvit'

num_unsup_concept = 0
num_spatial_concept = 0
num_heads = 8
attention_dropout = 0.1
projection_dropout = 0.1
ckpt_path = '/data/hqvo3/weights/CONCH/pytorch_model.bin' # for CONCH
num_workers = 4

# concept_embeddings_save_dir = 'datasets/GlomCrawl/embs/concepttrans_concept_embs_512.pth' # CONCH
concept_root = 'datasets/CUB/concepts/' # TODO: change datasets - relative path - manually generate `class2concepts.json`
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
concept_embeddings_save_dir = 'datasets/CUB/embs/concepttrans_eva_conceptori_embs_512_allconcepts.pth' # CONCH TODO: need to change if number of concepts changes
img_embeddings_save_dir = 'datasets/CUB/embs/concepttrans_eva_img_embs_512_aug0_iters1_{}.pth' # CONCH
labels_save_dir = 'datasets/CUB/embs/concepttrans_eva_labels_512_aug0_iters1_{}.pth' # CONCH

# concept_embeddings_save_dir = 'datasets/2_PanCancerTCGA/embs/concepttrans_vitL14_concept_embs_768.pth' # OPENCLIP
# img_embeddings_save_dir = 'datasets/2_PanCancerTCGA/embs/concepttrans_vitL14_img_embs_768_aug2_iters20_{}.pth' # OPENCLIP
# labels_save_dir = 'datasets/GlomCrawl/embs/concepttrans_vitL14_labels_768_aug2_iters20_{}.pth' # OPENCLIP

use_extracted_img_embs = True # TODO
only_visual_trunk = False # TODO
do_aug = True
n_iters = 1
aug_level = 0

# ----- only for learnable prompts -----
set_learnable_prompts = False
use_parent_level_learnable_prompts = False # whether to use 5 (parent) or 42 (child) soft prompts
csc = True # whether to use class-specific contexts or a general context, append class name to soft prompts

n_ctx_pos = 48
n_ctx_neg = 48
positive_prompt_init = ''
negative_prompt_init = ''
use_child_classes=True
use_parent_classes=False
use_attn_parent_branch=False

# --- weight balance for loss function ---
balanced_weights = None
balanced_weights_parents = None

# --- whether to extract patch-level embeddings for each input image (True) or unfreeze vision backbone (False) ---
use_patches_embeddings = True

# --- whether to use full concepts or partial 630 concept ---
use_full_concepts = False