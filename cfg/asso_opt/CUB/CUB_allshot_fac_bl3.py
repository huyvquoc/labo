_base_ = 'CUB_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/CUB/CUB_allshot_fac'
best_val_metric_name = 'val_main_cls_acc_multiclass_top1_micro' # for plotting only when model checkpoint
do_val_plotting=True
lr = 5e-5
bs = 512

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]

weight_decay = 0

concept_root = 'datasets/CUB/concepts_bl3/'
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'

balanced_weights = None
balanced_weights_parents = None