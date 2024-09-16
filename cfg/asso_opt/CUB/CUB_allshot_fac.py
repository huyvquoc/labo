_base_ = 'CUB_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/CUB/CUB_allshot_fac'
plotting_dir = 'output/asso_opt/figs' # for plotting only when model checkpoint
best_val_metric_name = 'val_main_cls_acc_multiclass_top1_micro' # for plotting only when model checkpoint
do_val_plotting=True
lr = 5e-5
bs = 512

concept_type = "all_submodular"
concept_select_fn = "submodular"
submodular_weights = [1e7, 0.1]

weight_decay = 0

balanced_weights = None
balanced_weights_parents = None 