_base_ = 'labo_base.py'
n_shots = "all"
data_root = 'exp/asso_opt_re/GlomCrawl/GlomCrawl_allshot_fac'
plotting_dir = 'output/concept_trans/figs_all_with_weight_balance_lsmoothing1' # for plotting only when model checkpoint
best_val_metric_name = 'val_main_cls_acc_multiclass_top1_micro' # for plotting only when model checkpoint
lr = 1e-2
bs = 1024

concept_type = "all_submodular"
# concept_select_fn = "submodular"
concept_select_fn = "clip"
use_mi = False
group_select = True
submodular_weights = [1e7, 1]
# submodular_weights = [1e9, 0.7]

# ----- only for ConceptTransformer -----
weight_decay = 0
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
balanced_weights = [0.6863153028201572, 2.8276190476190477, 0.9301378446115288, 0.6668912848158132, 2.0791316526610646, 3.7205513784461153, 0.4502578101304216, 5.049319727891157, \
                    0.8948161543098252, 0.3086920357662716, 1.0098639455782312, 1.5367494824016563, 2.5246598639455784, 0.6200918964076859, 5.049319727891157, 0.3086920357662716, \
                    0.728767795778105, 1.3594322344322345, 2.0197278911564625, 2.5246598639455784, 8.836309523809524, 0.5890873015873016, 1.4426627793974733, 4.158263305322129, \
                    3.5345238095238094, 1.4426627793974733, 0.3927248677248677, 0.21292312105565117, 0.9301378446115288, 0.728767795778105, 1.1781746031746032, 1.5367494824016563, \
                    3.7205513784461153, 0.9683626875407697, 1.472718253968254, 10.098639455782314, 2.8276190476190477, 0.9683626875407697, 5.049319727891157, 1.1588602654176425, \
                    8.836309523809524, 2.209077380952381]
balanced_weights_parents = [1.6779661016949152, 0.5981873111782477, 1.9223300970873787, 2.302325581395349, 0.5625]