import collections
import os

import numpy as np
from pathlib import Path
import pytorch_lightning as pl
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.classification import MulticlassAccuracy, MultilabelAccuracy
from transformers import get_cosine_schedule_with_warmup
import wandb

# from pytorch_lightning.pytorch.demos import Transformer
from loss.criterion import get_criterion

from metrics.cls import MetricsCalculator
import models.ctc
from models.ctc import concepts_cost, concepts_sparsity_cost, spatial_concepts_cost


def topk_argmax(tensor, k):
    # Get the top k values and their indices
    topk_values, topk_indices = torch.topk(tensor, k)

    # Sort the indices in descending order of values
    sorted_topk_indices = torch.argsort(topk_values, descending=True)

    # Get the top k argmax indices
    topk_argmax_indices = topk_indices[sorted_topk_indices]

    return topk_argmax_indices

class CTCModel(pl.LightningModule):

    def __init__(
        self, 
        cfg, # ----- LaBo from here -----
        class_names,
        select_idx=None,
        **kwargs, # ----- ConceptTransformer -----
    ) -> None:
        super().__init__()
        # ==================== LaBo starts here ====================
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        self.concept_root = Path(cfg.concept_root)

        if self.cfg.use_full_concepts:
            print('[CTCModel] Use full concepts')
            concept_raw_path = self.concept_root.joinpath('concepts_raw.npy')
            concept2cls_path = self.concept_root.joinpath('concept2cls.npy')
        else:
            print('[CTCModel] Use LaBo concepts')
            # concept_feat_path = self.data_root.joinpath('concepts_feat_{}.pth'.format(self.cfg.clip_model.replace('/','-'))) # --- NOT USED ---
            concept_raw_path = self.data_root.joinpath('concepts_raw_selected.npy')
            concept2cls_path = self.data_root.joinpath('concept2cls_selected.npy')
            select_idx_path = self.data_root.joinpath('select_idx.pth')

        suffix_work_dir = self.cfg.work_dir.split('/', 1)[-1]
        self.plotting_dir = ''.join(['output/', suffix_work_dir])

        # self.concepts = torch.load(concept_feat_path)[self.select_idx].cuda() # extracted text embeddings of all concepts
        # if self.cfg.use_txt_norm: self.concepts = self.concepts / self.concepts.norm(dim=-1, keepdim=True)

        if self.cfg.use_full_concepts:
            print('[CTCModel] Use full concepts')
            self.concept_raw = np.load(concept_raw_path) # all raw concepts (sentences)
            self.concept2cls = torch.from_numpy(np.load(concept2cls_path)).long().view(1, -1) # Mapping from a concept to its corresponding class
        else:
            if select_idx is None: self.select_idx = torch.load(select_idx_path)[:cfg.num_concept] # indices of all selected concepts
            else: self.select_idx = select_idx

            print('[CTCModel] Use LaBo concepts')
            self.concept_raw = np.load(concept_raw_path)[self.select_idx] # all raw concepts (sentences)
            self.concept2cls = torch.from_numpy(np.load(concept2cls_path)[self.select_idx]).long().view(1, -1) # Mapping from a concept to its corresponding class
        self.concept_raw = self.concept_raw.tolist()
        print('[CTCModel] Number of used concepts: ', self.concept2cls.shape[1])

        self.groupname2clsname = collections.OrderedDict(sorted(cfg.groupname2clsname.items()))
        self.groupnames = list(self.groupname2clsname.keys())

        if cfg.use_parent_level_learnable_prompts:
            self.num_cls_supp = len(self.groupnames)
        else:
            self.num_cls_supp = len(class_names)

        # ---------- FOR CHILD CLASSES ----------
        ### train ###
        self.train_perf_calc = MetricsCalculator(num_cls=cfg.num_cls, class_names=class_names, prefix_name='train_main_cls', report_top5=True, report_top7=False)
        ### val ###
        self.val_perf_calc = MetricsCalculator(num_cls=cfg.num_cls, class_names=class_names, prefix_name='val_main_cls', report_top5=True, report_top7=False)
        ### test ###
        self.test_perf_calc = MetricsCalculator(num_cls=cfg.num_cls, class_names=class_names, prefix_name='test_main_cls', report_top5=True, report_top7=False)
        # ----------------------------------------

        if cfg.use_parent_classes:
            # ---------- FOR PARENT CLASSES ----------
            ### train ###
            self.train_group_perf_calc = MetricsCalculator(num_cls=len(self.groupnames), class_names=self.groupnames, prefix_name='train_sup_cls', report_top5=True, report_top7=False)
            ### val ###
            self.val_group_perf_calc = MetricsCalculator(num_cls=len(self.groupnames), class_names=self.groupnames, prefix_name='val_sup_cls', report_top5=True, report_top7=False)
            ### test ###
            self.test_group_perf_calc = MetricsCalculator(num_cls=len(self.groupnames), class_names=self.groupnames, prefix_name='test_sup_cls', report_top5=True, report_top7=False)
            # ----------------------------------------

        # ---------- FOR QK OF PARENT CLASSES ----------
        if cfg.use_parent_level_learnable_prompts:
            self.class_names_supp = self.groupnames
        else:
            self.class_names_supp = class_names

        ### train ###
        self.train_qk_group_perf_calc = MetricsCalculator(num_cls=self.num_cls_supp, class_names=self.class_names_supp, prefix_name='train_qk_hierarchy', report_top5=True, report_top7=False)
        ### val ###
        self.val_qk_group_perf_calc = MetricsCalculator(num_cls=self.num_cls_supp, class_names=self.class_names_supp, prefix_name='val_qk_hierarchy', report_top5=True, report_top7=False)
        ### test ###
        self.test_qk_group_perf_calc = MetricsCalculator(num_cls=self.num_cls_supp, class_names=self.class_names_supp, prefix_name='test_qk_hierarchy', report_top5=True, report_top7=False)
        # ----------------------------------------

        self.best_val_metric = None

        self.train_step_outputs = []
        self.train_step_targets = []
        self.train_step_outputs_group = []
        self.train_step_targets_group = []
        self.train_step_outputs_qk_group = []
        self.train_step_targets_qk_group = []

        self.val_step_outputs = []
        self.val_step_targets = []
        self.val_step_outputs_group = []
        self.val_step_targets_group = []
        self.val_step_outputs_qk_group = []
        self.val_step_targets_qk_group = []

        self.test_step_outputs = []
        self.test_step_targets = []
        self.test_step_outputs_group = []
        self.test_step_targets_group = []
        self.test_step_outputs_qk_group = []
        self.test_step_targets_qk_group = []

        # self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.cfg.num_cls)
        self.balanced_weights = cfg.balanced_weights
        if self.balanced_weights is not None:
            self.balanced_weights = torch.DoubleTensor(self.balanced_weights).cuda()

        self.balanced_weights_parents = cfg.balanced_weights_parents
        if self.balanced_weights_parents is not None:
            self.balanced_weights_parents = torch.DoubleTensor(self.balanced_weights_parents).cuda()

        self.criterion = get_criterion(cfg)
        self.criterion.cuda()
        self.save_hyperparameters()
        # ==================== LaBo ends here ====================
        # ==================== ConceptTransformer starts here ====================
        # self.save_hyperparameters() # comment this because it has already been at the last line of LaBo

        # Instantiate CTC model
        self.model = getattr(models.ctc, self.cfg.ctc_model)(self.cfg, class_names, self.concept2cls) # self.model is CTC
        # ctc_model='mnist_ctc' for MNIST odd/even
        # ==================== ConceptTransformer ends here ====================
        self.first_val_complete = False

    def forward(self, x):
        '''
        Inputs:
        - x: can either be original images with shape (train_batch_size, #channels, width, height)
        OR extrated embeddings with shape (train_batch_size, embedding_dim)
        '''
        # ==================== LaBo starts here ====================
        # ==================== LaBo ends here ====================
        # ==================== ConceptTransformer starts here ====================
        '''
        TODO: we do not have x here. x is an original image, not image embedding. Because we extracted image embeddings using a separate process and save them into a pth file
        x: torch.Tensor (batch, 1, img_width, img_height), (32, 1, 28, 28) for MNIST odd/even, (32, 3, 224, 224) for CUB200
        TODO: Maybe we can just feed the features directly to the network, not need the original image (see forward of CVIT, original x is not used anymore after the feature extraction step).
        But one problem is 512 vs 1024, ConceptTransformer also has predefined cls_token from the pretrained feature extractor
        TODO: We should feed original image because later we will need to implement soft prompting, which means we will not have a fix collection of input
        '''
        outputs, outputs_group, unsup_concept_attn, concept_attn, spatial_concept_attn, qk_parent_class = self.model(x, self.concept_raw) # self.model is CTC for MNIST odd/even, and CVIT for CUB

        logits = None
        if outputs is not None:
            logits = F.log_softmax(outputs, dim=-1)

        logits_group = None
        if outputs_group is not None:
            logits_group = F.log_softmax(outputs_group, dim=-1) # can be None

        logits_qk_parent_class = None
        if qk_parent_class is not None:
            logits_qk_parent_class = F.log_softmax(qk_parent_class, dim=-1)

        return logits, logits_group, unsup_concept_attn, concept_attn, spatial_concept_attn, logits_qk_parent_class
        # ==================== ConceptTransformer ends here ====================


    def training_step(self, train_batch, batch_idx):
        # ==================== ConceptTransformer starts here ====================
        '''
        - For CUB200
        len(batch): 4
        batch[0].shape (original images): (32, 3, 224, 224)
        batch[1].shape (global contexts attn matrix): (32, 13)
        batch[2].shape (local contexts attn matrix of patches and contexts): (32, 196, 95)
        batch[3].shape (labels): (32)

        TODO: will keep mostly the ConceptTransformer code
        need to check if the required `batch` below matches with the argument `train_batch`
        '''
        logits, preds, y, cls_loss, cls_loss_group, cls_loss_qk_group, expl_loss, l1_loss, _, _, _ = self.shared_step(train_batch, stage='train')
        # logits.shape: (32, 200)
        # preds.shape: (32)
        loss = cls_loss
        if cls_loss_group is not None:
            loss = loss + self.cfg.group_lambda * cls_loss_group
        if cls_loss_qk_group is not None:
            loss = loss + self.cfg.qk_group_lambda * cls_loss_qk_group
        if self.cfg.expl_lambda > 0:
            loss = loss + self.cfg.expl_lambda * expl_loss
        if self.cfg.attention_sparsity > 0:
            loss = loss + self.cfg.attention_sparsity * l1_loss

        # merge 3 dicts
        # all_metrics_dict = {**perf_dict, **group_perf_dict, **qk_group_perf_dict}
        # all_metrics_dict = {**perf_dict, **qk_group_perf_dict}
        # add losses to the new merged dict

        loss_dict = {
            "train_loss": loss,
            "train_main_loss": cls_loss,
            # "train_ce_loss_group": cls_loss_group,
            "train_qk_loss_group": cls_loss_qk_group,
            "train_expl_loss": expl_loss,
            "train_l1_loss": l1_loss,
        }

        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
        )
        return loss
        # ==================== ConceptTransformer ends here ====================
    
    def shared_step(self, batch, stage):
        '''
        `shared_step` only exists in ConceptTransformer -> No need to modify it
        '''
        # ==================== ConceptTransformer starts here ====================
        # MNIST: x.shape: (32, 1, 28, 28), expl.shape: (32, 10), spatial_expl.shape: (32), y.shape: (32)
        # CUB: x.shape: (32, 3, 224, 224), expl.shape: (32, 13), spatial_expl.shape: (32, 196, 95), y.shape: (32)
        x, expl, spatial_expl, y, y_group = batch # TODO: data module needs return y_group
        # MNIST:
        # CUB: logits.shape: (32, 200), unsup_concept_attn.shape: None, concept_attn.shape: (32, 1, 13), spatial_concept_attn.shape: (32, 196, 95)
        logits, logits_group, unsup_concept_attn, concept_attn, spatial_concept_attn, logits_qk_parent_class = self(x)
        # logits = F.softmax(logits)

        if logits is not None:
            preds = torch.argmax(logits, dim=1) # shape: (32)
        else:
            preds = None

        # ----- loss functions from LaBo -----
        # cls_loss = F.cross_entropy(pred, F.one_hot(label, num_classes=self.cfg.num_cls).float(), weight=self.balanced_weights)
        # cls_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(label, num_classes=self.cfg.num_cls).float(), weight=self.balanced_weights)
        # cls_loss, _ = self.criterion(pred, F.one_hot(label, num_classes=self.cfg.num_cls).float())
        # ----- loss functions from ConceptTransformer -----
        # cls_loss = F.nll_loss(logits, y) # MNIST: logits.shape: (32, 2), y.shape: (32); CUB: logits.shape: (32, 200), y.shape: (32)
        # cls_loss = F.cross_entropy(logits, F.one_hot(y, num_classes=self.cfg.num_cls).float(), weight=self.balanced_weights)
        # cls_loss = F.binary_cross_entropy_with_logits(logits, F.one_hot(y, num_classes=self.cfg.num_cls).float())
        # cls_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(label, num_classes=self.cfg.num_cls).float(), weight=self.balanced_weights)

        # ----- USE CHILD CLASSES -----
        if logits is not None:
            cls_loss = F.cross_entropy(logits, F.one_hot(y, num_classes=self.cfg.num_cls).float(), weight=self.balanced_weights)
            # cls_loss, reg_loss = self.criterion(logits, F.one_hot(y, num_classes=self.cfg.num_cls).float()) # for ANLS
            # cls_loss = self.criterion(logits, F.one_hot(y, num_classes=self.cfg.num_cls).float())
        else:
            cls_loss = 0
        # -----------------------------

        # ----- USE PARENT CLASSES -----
        if logits_group is not None:
            cls_loss_group = F.cross_entropy(logits_group, F.one_hot(y_group, num_classes=logits_group.shape[1]).float(), weight=self.balanced_weights_parents)
            # cls_loss_group, reg_loss_group = self.criterion(logits_group, F.one_hot(y_group, num_classes=logits_group.shape[1]).float()) # for ANLS
            # cls_loss_group = F.nll_loss(logits_group, y_group) # MNIST: logits.shape: (32, 2), y.shape: (32); CUB: logits.shape: (32, 200), y.shape: (32)
        else:
            cls_loss_group = 0
        # -----------------------------

        # ----- USE QK OF PARENT CLASSES -----
        if logits_qk_parent_class is not None:
            cls_loss_qk_group = F.cross_entropy(logits_qk_parent_class, F.one_hot(y_group, num_classes=logits_qk_parent_class.shape[1]).float(), weight=self.balanced_weights_parents)
            # cls_loss_qk_group, reg_loss_qk_group = self.criterion(logits_qk_parent_class, F.one_hot(y_group, num_classes=logits_qk_parent_class.shape[1]).float()) # for ANLS
            # cls_loss_qk_group = F.nll_loss(logits_qk_parent_class, y_group) # MNIST: logits.shape: (32, 2), y.shape: (32); CUB: logits.shape: (32, 200), y.shape: (32)
        else:
            cls_loss_qk_group = 0
        # -----------------------------

        if stage == 'train':
            self.train_step_outputs.append(logits)
            self.train_step_targets.append(y)
            self.train_step_outputs_group.append(logits_group)
            self.train_step_targets_group.append(y_group)
            self.train_step_outputs_qk_group.append(logits_qk_parent_class)
            self.train_step_targets_qk_group.append(y_group)
        elif stage == 'val':
            self.val_step_outputs.append(logits)
            self.val_step_targets.append(y)
            self.val_step_outputs_group.append(logits_group)
            self.val_step_targets_group.append(y_group)
            self.val_step_outputs_qk_group.append(logits_qk_parent_class)
            self.val_step_targets_qk_group.append(y_group)
        elif stage == 'test':
            self.test_step_outputs.append(logits)
            self.test_step_targets.append(y)
            self.test_step_outputs_group.append(logits_group)
            self.test_step_targets_group.append(y_group)
            self.test_step_outputs_qk_group.append(logits_qk_parent_class)
            self.test_step_targets_qk_group.append(y_group)

        expl_loss = concepts_cost(concept_attn, expl) + spatial_concepts_cost( # PRED concept_attn.shape: (32, 1, 13), TARGET expl.shape: (32, 13); for glom crawl concept_attn.shape (990, 1, 630) and expl.shape (990, 630)
            spatial_concept_attn, spatial_expl # PRED spatial_concept_attn.shape: (32, 196, 95), TARGET spatial_expl.shpae: (32, 196, 95)
        )
        l1_loss = concepts_sparsity_cost(concept_attn, spatial_concept_attn)

        return (
            logits,
            preds,
            y,
            cls_loss,
            cls_loss_group,
            cls_loss_qk_group,
            expl_loss,
            l1_loss,
            unsup_concept_attn,
            concept_attn,
            spatial_concept_attn,
        )
        # ==================== ConceptTransformer ends here ====================

    def configure_optimizers(self):
        # ==================== LaBo starts here ====================
        # opt = th.optim.Adam(self.parameters(), lr=self.cfg.lr)
        # return opt
        # ==================== LaBo ends here ====================
        # ==================== ConceptTransformer starts here ====================
        '''
        TODO: in LaBo has only optimizer, in ConceptTransformer has both optimizer and scheduler.
        Check if keeping only ConceptTransformer will work normally
        '''
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        if self.cfg.disable_lr_scheduler:
            return [optimizer]
        else:
            # scheduler = LinearWarmupCosineAnnealingLR(
            #     optimizer, warmup_epochs=self.cfg.warmup, max_epochs=self.cfg.max_epochs
            # )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=self.cfg.warmup, T_mult=1, eta_min=1e-6, last_epoch=-1
            )
            # scheduler = get_cosine_schedule_with_warmup(
            #     optimizer, num_warmup_steps=self.cfg.warmup, num_training_steps=20000
            # )
        return [optimizer], [scheduler]
        # ==================== ConceptTransformer ends here ====================


    def validation_step(self, batch, batch_idx):
        # ==================== LaBo starts here ====================
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric(self.cfg.best_val_metric_name, summary='max')
        # ==================== LaBo ends here ====================
        # ==================== ConceptTransformer starts here ====================
        '''
        TODO: Both `validation_step` have the same arguments `batch` and `batch_idx`. Both do not use batch_idx
        Need to check if the required `batch` is actually the argument `batch`
        '''
        logits, preds, y, cls_loss, cls_loss_group, cls_loss_qk_group, expl_loss, l1_loss, _, _, _ = self.shared_step(batch, stage='val')

        loss_dict = {
            "val_ce_loss": cls_loss,
            # "val_ce_loss_group": cls_loss_group,
            "val_qk_loss_group": cls_loss_qk_group,
            "val_expl_loss": expl_loss,
            "val_l1_loss": l1_loss,
        }

        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
        )

        return cls_loss
        # return all_metrics_dict['val_main_cls_acc_multiclass_top1_micro']
        # return ce_loss # LaBo


    def test_step(self, batch, batch_idx):
        # ==================== ConceptTransformer starts here ====================
        '''
        TODO: do we need to append y and pred into overall lists?
        '''
        logits, preds, y, cls_loss, cls_loss_group, cls_loss_qk_group, expl_loss, l1_loss, _, concept_attn, _ = self.shared_step(batch, stage='test')

        for sample_id, cur_label in enumerate(y):
            if sample_id == 5:
                break
            print('[in testing] cur_label: ', cur_label)
            s_concept_id, e_concept_id = cur_label*50, (cur_label + 1)*50 - 1
            print('[in testing] type(concept_attn): ', type(concept_attn))
            print('[in testing] concept_attn.shape: ', concept_attn.shape)
            top_concept_ids = topk_argmax(concept_attn[sample_id, 0], 50)
            print('[in testing] top 50 concept id predicted by ConceptTransformer: ', top_concept_ids)

            count_satisfied_concepts = ((top_concept_ids >= s_concept_id) & (top_concept_ids <= e_concept_id)).sum()
            print('=> Count satisfied concepts: ', count_satisfied_concepts)
            print('[in testing] y[0]: ', cur_label)
            print(f'[in testing] Range of concept ids of current class: start {s_concept_id} - end {e_concept_id}')

        loss_dict = {
            "test_ce_loss": cls_loss,
            # "test_ce_loss_group": cls_loss_group,
            "test_qk_loss_group": cls_loss_qk_group,
            "test_expl_loss": expl_loss,
            "test_l1_loss": l1_loss,
        }

        self.log_dict(
            loss_dict
        )

        return cls_loss
        # return all_metrics_dict['test_main_cls_acc_multiclass_top1_micro']
        # return loss # LaBo
        # ==================== ConceptTransformer ends here ====================

    def on_train_epoch_end(self) :
        train_all_outputs = torch.cat(self.train_step_outputs, dim=0)
        train_all_targets = torch.cat(self.train_step_targets, dim=0)
        train_all_outputs_group = torch.cat(self.train_step_outputs_group, dim=0) if self.train_step_outputs_group[0] is not None else None
        train_all_targets_group = torch.cat(self.train_step_targets_group, dim=0)
        train_all_outputs_qk_group = torch.cat(self.train_step_outputs_qk_group, dim=0) if self.train_step_outputs_qk_group[0] is not None else None
        train_all_targets_qk_group = torch.cat(self.train_step_targets_qk_group, dim=0)

        perf_dict = self.train_perf_calc.generate_dict_of_metrics(train_all_outputs, train_all_targets)
        # group_perf_dict = self.train_group_perf_calc.generate_dict_of_metrics(logits_group, y_group)
        qk_group_perf_dict = self.train_qk_group_perf_calc.generate_dict_of_metrics(train_all_outputs_qk_group, train_all_targets_qk_group)

        # merge 3 dicts
        # all_metrics_dict = {**perf_dict, **group_perf_dict, **qk_group_perf_dict}
        all_metrics_dict = {**perf_dict, **qk_group_perf_dict}
        self.log_dict(
            all_metrics_dict,
            on_step=False,
            on_epoch=True,
        )

        # clear rem lists
        self.train_step_outputs.clear()
        self.train_step_targets.clear()
        self.train_step_outputs_group.clear()
        self.train_step_targets_group.clear()
        self.train_step_outputs_qk_group.clear()
        self.train_step_targets_qk_group.clear()

    def on_validation_epoch_end(self):
        val_all_outputs = torch.cat(self.val_step_outputs, dim=0)
        val_all_targets = torch.cat(self.val_step_targets, dim=0)
        val_all_outputs_group = torch.cat(self.val_step_outputs_group, dim=0) if self.val_step_outputs_group[0] is not None else None
        val_all_targets_group = torch.cat(self.val_step_targets_group, dim=0)
        val_all_outputs_qk_group = torch.cat(self.val_step_outputs_qk_group, dim=0) if self.val_step_outputs_qk_group[0] is not None else None
        val_all_targets_qk_group = torch.cat(self.val_step_targets_qk_group, dim=0)

        perf_dict = self.val_perf_calc.generate_dict_of_metrics(val_all_outputs, val_all_targets)
        # group_perf_dict = self.val_group_perf_calc.generate_dict_of_metrics(logits_group, y_group)
        qk_group_perf_dict = self.val_qk_group_perf_calc.generate_dict_of_metrics(val_all_outputs_qk_group, val_all_targets_qk_group)

        # merge 3 dicts
        # all_metrics_dict = {**perf_dict, **group_perf_dict, **qk_group_perf_dict}
        all_metrics_dict = {**perf_dict, **qk_group_perf_dict}
        self.log_dict(
            all_metrics_dict,
            on_step=False,
            on_epoch=True,
        )

        if self.first_val_complete and self.cfg.do_val_plotting:
            # plotting
            if self.best_val_metric is None or perf_dict[self.cfg.best_val_metric_name] >= self.best_val_metric:
                os.makedirs(self.plotting_dir, exist_ok=True)
                if len(os.listdir(self.plotting_dir)) != 0:
                    for old_plot in os.listdir(self.plotting_dir):
                        old_plot_path = os.path.join(self.plotting_dir, old_plot)
                        os.unlink(old_plot_path)

                val_metric_info = ''.join([self.cfg.best_val_metric_name, '_', str(perf_dict[self.cfg.best_val_metric_name].item())])
                print(f'[**] Found new best val metric !! - {val_metric_info}')
                self.val_perf_calc.generate_plots(val_all_outputs, val_all_targets, save_dir=self.plotting_dir, suffix=val_metric_info)
                self.val_qk_group_perf_calc.generate_plots(val_all_outputs_qk_group, val_all_targets_qk_group, save_dir=self.plotting_dir, suffix=val_metric_info)

                self.best_val_metric = perf_dict[self.cfg.best_val_metric_name]
        else:
            self.first_val_complete = True

        # clear rem lists
        self.val_step_outputs.clear()
        self.val_step_targets.clear()
        self.val_step_outputs_group.clear()
        self.val_step_targets_group.clear()
        self.val_step_outputs_qk_group.clear()
        self.val_step_targets_qk_group.clear()
    
    def on_test_epoch_end(self):
        test_all_outputs = torch.cat(self.test_step_outputs, dim=0)
        test_all_targets = torch.cat(self.test_step_targets, dim=0)
        test_all_outputs_group = torch.cat(self.test_step_outputs_group, dim=0) if self.test_step_outputs_group[0] is not None else None
        test_all_targets_group = torch.cat(self.test_step_targets_group, dim=0)
        test_all_outputs_qk_group = torch.cat(self.test_step_outputs_qk_group, dim=0) if self.test_step_outputs_qk_group[0] is not None else None
        test_all_targets_qk_group = torch.cat(self.test_step_targets_qk_group, dim=0)

        perf_dict = self.test_perf_calc.generate_dict_of_metrics(test_all_outputs, test_all_targets)
        # group_perf_dict = self.test_group_perf_calc.generate_dict_of_metrics(logits_group, y_group)
        qk_group_perf_dict = self.test_qk_group_perf_calc.generate_dict_of_metrics(test_all_outputs_qk_group, test_all_targets_qk_group)

        # merge 3 dicts
        # all_metrics_dict = {**perf_dict, **group_perf_dict, **qk_group_perf_dict}
        all_metrics_dict = {**perf_dict, **qk_group_perf_dict}
        self.log_dict(
            all_metrics_dict,
            on_step=False,
            on_epoch=True,
        )

        os.makedirs(self.plotting_dir, exist_ok=True)
        # if len(os.listdir(self.plotting_dir)) != 0:
        #     for old_plot in os.listdir(self.plotting_dir):
        #         old_plot_path = os.path.join(self.plotting_dir, old_plot)
        #         os.unlink(old_plot_path)

        test_metric_info = 'test'
        self.test_perf_calc.generate_plots(test_all_outputs, test_all_targets, save_dir=self.plotting_dir, suffix=test_metric_info)
        self.test_qk_group_perf_calc.generate_plots(test_all_outputs_qk_group, test_all_targets_qk_group, save_dir=self.plotting_dir, suffix=test_metric_info)

        # clear rem lists
        self.test_step_outputs.clear()
        self.test_step_targets.clear()
        self.test_step_outputs_group.clear()
        self.test_step_targets_group.clear()
        self.test_step_outputs_qk_group.clear()
        self.test_step_targets_qk_group.clear()

    # def test_epoch_end(self, outputs):
    #     # ==================== LaBo starts here ====================
    #     # all_y = torch.hstack(self.all_y)
    #     # all_pred = torch.hstack(self.all_pred)
    #     # all_logit = torch.hstack(self.all_logit)
    #     # self.total_test_acc = self.test_acc(all_logit, all_y)
    #     # self.total_test_f1 = self.test_f1(all_pred, all_y)
    #     # self.total_test_ap = self.test_ap(all_logit, all_y)
    #     pass
    #     # ==================== LaBo ends here ====================
    