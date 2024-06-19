_base_ = 'GlomCrawl_base.py'
n_shots = "all"
data_root = 'exp/asso_opt/GlomCrawl/GlomCrawl_allshot_fac'
lr = 1e-4
bs = 1024

concept_type = "all_submodular"
# concept_select_fn = "submodular"
concept_select_fn = "clip"
use_mi = False
group_select = True
submodular_weights = [1e7, 1]
# submodular_weights = [1e9, 0.7]