_base_ = 'PanCancerTCGA_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/2_PanCancerTCGA/PanCancerTCGA_allshot_fac'
best_val_metric_name = 'val_main_cls_acc_multiclass_top1_macro' # for plotting only when model checkpoint
do_val_plotting=True
lr = 1e-3
bs = 2048

concept_type = "all_submodular"
# concept_select_fn = "submodular"
concept_select_fn = "clip"
use_mi = False
group_select = True
submodular_weights = [1e7, 1]
# submodular_weights = [1e9, 0.7]

# ----- only for ConceptTransformer -----
weight_decay = 1e-10
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
concept_embeddings_save_dir = 'datasets/2_PanCancerTCGA/embs/concepttrans_concept_embs_512_allconcepts_mpe.pth' # CONCH TODO: need to change if number of concepts changes
img_embeddings_save_dir = 'datasets/2_PanCancerTCGA/embs/concepttrans_img_embs_512_aug0_iters1_mpe_{}.pth' # CONCH
labels_save_dir = 'datasets/2_PanCancerTCGA/embs/concepttrans_labels_512_aug0_iters1_mpe_{}.pth' # CONCH

# concept_embeddings_save_dir = 'datasets/2_PanCancerTCGA/embs/concepttrans_vitL14_concept_embs_768.pth' # OPENCLIP
# img_embeddings_save_dir = 'datasets/2_PanCancerTCGA/embs/concepttrans_vitL14_img_embs_768_aug2_iters20_{}.pth' # OPENCLIP
# labels_save_dir = 'datasets/GlomCrawl/embs/concepttrans_vitL14_labels_768_aug2_iters20_{}.pth' # OPENCLIP

use_extracted_img_embs = True # TODO
only_visual_trunk = True # TODO
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
use_attn_parent_branch=True

# --- weight balance for loss function ---
balanced_weights = [1.7184551886792452, 0.8433159722222222, 0.3505251763951251, 0.3592825443786982, 1.3493055555555555, 9.587171052631579, 1.040892857142857, 2.5299479166666665, 0.35508040935672514, 0.7335151006711409, 3.4369103773584904, 0.7354895693135935, 1.2858088235294118, 1.0291313559322033, 0.5145656779661016, 0.49860287408759124, 10.119791666666666, 3.8756648936170213, 3.373263888888889, 2.031482342007435, 6.505580357142857, 0.8771569020866774, 4.592174369747899, 0.6302984429065744, 0.855193661971831, 0.891466150081566, 1.366171875, 2.802403846153846, 0.7496141975308642, 4.047916666666667, 0.6796875, 5.254507211538462]
balanced_weights_parents = [1.7184551886792452, 0.8433159722222222, 0.3505251763951251, 0.3592825443786982, 1.3493055555555555, 9.587171052631579, 1.040892857142857, 2.5299479166666665, 0.35508040935672514, 0.7335151006711409, 3.4369103773584904, 0.7354895693135935, 1.2858088235294118, 1.0291313559322033, 0.5145656779661016, 0.49860287408759124, 10.119791666666666, 3.8756648936170213, 3.373263888888889, 2.031482342007435, 6.505580357142857, 0.8771569020866774, 4.592174369747899, 0.6302984429065744, 0.855193661971831, 0.891466150081566, 1.366171875, 2.802403846153846, 0.7496141975308642, 4.047916666666667, 0.6796875, 5.254507211538462]

# --- whether to extract patch-level embeddings for each input image (True) or unfreeze vision backbone (False) ---
use_patches_embeddings = True

# --- whether to use full concepts or partial 630 concept ---
use_full_concepts = True