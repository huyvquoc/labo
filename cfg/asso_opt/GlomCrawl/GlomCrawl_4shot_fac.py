_base_ = 'GlomCrawl_base.py'
n_shots = 4
data_root = 'exp/asso_opt/GlomCrawl/GlomCrawl_4shot_fac'
lr = 1e-5
bs = 64

concept_type = "all_submodular"
concept_select_fn = "clip"
use_mi = False
group_select = True
submodular_weights = [1e7, 10]