_base_ = 'GlomCrawl_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/GlomCrawl/GlomCrawl_allshot_fac'
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
weight_decay = 1e-7
disable_lr_scheduler = True
warmup = 100
expl_lambda = 2.0
attention_sparsity = 0.0
ctc_model = 'cub_cvit'

num_unsup_concept = 0
num_spatial_concept = 0
num_heads = 8
attention_dropout = 0.1
projection_dropout = 0.1
ckpt_path = '/data/hqvo3/weights/CONCH/pytorch_model.bin' # for CONCH
num_workers = 4
concept_embeddings_save_dir = 'datasets/GlomCrawl/embs/concepttrans_concept_embs_768.pth' # TODO
img_embeddings_save_dir = 'datasets/GlomCrawl/embs/concepttrans_img_embs_768_aug3_{}.pth' # TODO
labels_save_dir = 'datasets/GlomCrawl/embs/concepttrans_labels_768_aug3_{}.pth'
use_extracted_img_embs = True # TODO
only_visual_trunk = True # TODO
do_aug = True
n_iters = 5
aug_level = 3