_base_ = 'GlomCrawl_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/GlomCrawl/GlomCrawl_allshot_fac'
plotting_dir = 'output/concept_trans/figs_all_without_weight_balance_lsmoothing2_rerun_highest' # for plotting only when model checkpoint
best_val_metric_name = 'val_main_cls_acc_multiclass_top1_micro' # for plotting only when model checkpoint
do_val_plotting=False
lr = 1e-2
bs = 4096

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
warmup = 1000
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
# concept_embeddings_save_dir = 'datasets/GlomCrawl/embs/concepttrans_concept_embs_768.pth' # TODO
# concept_embeddings_save_dir = 'datasets/GlomCrawl/embs/concepttrans_concept_embs_768_debug.pth'
# img_embeddings_save_dir = 'datasets/GlomCrawl/embs/concepttrans_img_embs_768_aug2_iters1_{}.pth' # TODO
# labels_save_dir = 'datasets/GlomCrawl/embs/concepttrans_labels_768_aug2_iters1_{}.pth'
concept_embeddings_save_dir = 'datasets/GlomCrawl/embs/concepttrans_concept_embs_512.pth' # TODO
img_embeddings_save_dir = 'datasets/GlomCrawl/embs/concepttrans_img_embs_512_aug2_iters20_{}.pth' # TODO
labels_save_dir = 'datasets/GlomCrawl/embs/concepttrans_labels_512_aug2_iters20_{}.pth'
# img_embeddings_save_dir = 'datasets/GlomCrawl/embs/concepttrans_img_embs_512_noaug_{}.pth' # TODO
# labels_save_dir = 'datasets/GlomCrawl/embs/concepttrans_labels_512_noaug_{}.pth'
use_extracted_img_embs = True # TODO
only_visual_trunk = False # TODO
do_aug = True
n_iters = 20
aug_level = 2

# ----- only for learnable prompts -----
set_learnable_prompts = True
csc = True # whether to use class-specific contexts or a general context
n_ctx_pos = 48
n_ctx_neg = 48
positive_prompt_init = ''
negative_prompt_init = ''
use_child_classes=True
use_parent_classes=False
use_attn_parent_branch=True

# --- weight balance for loss function ---
balanced_weights = None
balanced_weights_parents = None

# --- whether to extract patch-level embeddings for each input image (True) or unfreeze vision backbone (False) ---
use_patches_embeddings = False