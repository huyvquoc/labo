import collections

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
        concept_feat_path = self.data_root.joinpath('concepts_feat_{}.pth'.format(self.cfg.clip_model.replace('/','-')))
        concept_raw_path = self.data_root.joinpath('concepts_raw_selected.npy')
        concept2cls_path = self.data_root.joinpath('concept2cls_selected.npy')
        select_idx_path = self.data_root.joinpath('select_idx.pth')

        if not concept_feat_path.exists():
            raise RuntimeError('need to call datamodule precompute_txt before using the model')
        else:
            if select_idx is None: self.select_idx = torch.load(select_idx_path)[:cfg.num_concept] # indices of all selected concepts
            else: self.select_idx = select_idx

            self.concepts = torch.load(concept_feat_path)[self.select_idx].cuda() # extracted text embeddings of all concepts
            if self.cfg.use_txt_norm: self.concepts = self.concepts / self.concepts.norm(dim=-1, keepdim=True)

            self.concept_raw = np.load(concept_raw_path)[self.select_idx] # all raw concepts (sentences)
            self.concept_raw = self.concept_raw.tolist()
            self.concept2cls = torch.from_numpy(np.load(concept2cls_path)[self.select_idx]).long().view(1, -1) # Mapping from a concept to its corresponding class

        self.groupname2clsname = collections.OrderedDict(sorted(cfg.groupname2clsname.items()))
        self.groupnames = list(self.groupname2clsname.keys())

        # ---------- FOR CHILD CLASSES ----------
        ### train ###
        self.train_perf_calc = MetricsCalculator(num_cls=cfg.num_cls, class_names=class_names, prefix_name='train_main_cls')
        ### val ###
        self.val_perf_calc = MetricsCalculator(num_cls=cfg.num_cls, class_names=class_names, prefix_name='val_main_cls')
        ### test ###
        self.test_perf_calc = MetricsCalculator(num_cls=cfg.num_cls, class_names=class_names, prefix_name='test_main_cls')
        # ----------------------------------------

        # ---------- FOR PARENT CLASSES ----------
        ### train ###
        self.train_group_perf_calc = MetricsCalculator(num_cls=cfg.num_cls_group, class_names=self.groupnames, prefix_name='train_sup_cls')
        ### val ###
        self.val_group_perf_calc = MetricsCalculator(num_cls=cfg.num_cls_group, class_names=self.groupnames, prefix_name='val_sup_cls')
        ### test ###
        self.test_group_perf_calc = MetricsCalculator(num_cls=cfg.num_cls_group, class_names=self.groupnames, prefix_name='test_sup_cls')
        # ----------------------------------------

        # ---------- FOR QK OF PARENT CLASSES ----------
        ### train ###
        self.train_qk_group_perf_calc = MetricsCalculator(num_cls=cfg.num_cls_group, class_names=self.groupnames, prefix_name='train_qk_hierarchy')
        ### val ###
        self.val_qk_group_perf_calc = MetricsCalculator(num_cls=cfg.num_cls_group, class_names=self.groupnames, prefix_name='val_qk_hierarchy')
        ### test ###
        self.test_qk_group_perf_calc = MetricsCalculator(num_cls=cfg.num_cls_group, class_names=self.groupnames, prefix_name='test_qk_hierarchy')
        # ----------------------------------------

        self.all_y = []
        self.all_pred = []
        self.all_logit = []
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.cfg.num_cls)
        self.balanced_weights = torch.DoubleTensor(cfg.balanced_weights).cuda()
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

    def forward(self, x):
        '''
        Inputs:
        - x: can either be original images with shape (train_batch_size, #channels, width, height)
        OR extrated embeddings with shape (train_batch_size, embedding_dim)
        '''
        # ==================== LaBo starts here ====================
        # self.concepts # (#selected_concepts, embedding_dim) = (630, 512)
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
        logits, preds, cls_loss, cls_loss_group, cls_loss_qk_group, perf_dict, group_perf_dict, qk_group_perf_dict, expl_loss, l1_loss, _, _, _ = self.shared_step(train_batch, stage='train')
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

        # TODO: merge 3 dicts
        perf_dict, group_perf_dict, qk_group_perf_dict
        # TODO: add losses to the new merged dict

        self.log_dict(
            {
                "train_loss": loss,
                "train_ce_loss": cls_loss,
                # "train_ce_loss_group": cls_loss_group,
                "train_qk_loss_group": cls_loss_qk_group,
                "train_acc": acc,
                # "train_acc_group": acc_group,
                "train_acc_qk_group": acc_qk_group,
                "train_expl_loss": expl_loss,
                "train_l1_loss": l1_loss,
            },
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
        # cls_loss = F.cross_entropy(pred, F.one_hot(label, num_classes=42).float(), weight=self.balanced_weights)
        # cls_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(label, num_classes=42).float(), weight=self.balanced_weights)
        # cls_loss, _ = self.criterion(pred, F.one_hot(label, num_classes=42).float())
        # ----- loss functions from ConceptTransformer -----
        # cls_loss = F.nll_loss(logits, y) # MNIST: logits.shape: (32, 2), y.shape: (32); CUB: logits.shape: (32, 200), y.shape: (32)
        # cls_loss = F.cross_entropy(logits, F.one_hot(y, num_classes=42).float(), weight=self.balanced_weights)
        # cls_loss = F.binary_cross_entropy_with_logits(logits, F.one_hot(y, num_classes=42).float())
        # cls_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(label, num_classes=42).float(), weight=self.balanced_weights)

        # ----- USE CHILD CLASSES -----
        if logits is not None:
            cls_loss, reg_loss = self.criterion(logits, F.one_hot(y, num_classes=42).float()) # for ANLS
            # cls_loss = self.criterion(logits, F.one_hot(y, num_classes=42).float())
        else:
            cls_loss = 0
        # -----------------------------

        # ----- USE PARENT CLASSES -----
        if logits_group is not None:
            # cls_loss_group = F.nll_loss(logits_group, y_group) # MNIST: logits.shape: (32, 2), y.shape: (32); CUB: logits.shape: (32, 200), y.shape: (32)
            cls_loss_group, reg_loss_group = self.criterion(logits_group, F.one_hot(y_group, num_classes=5).float()) # for ANLS
        else:
            cls_loss_group = 0
        # -----------------------------

        # ----- USE QK OF PARENT CLASSES -----
        if logits_qk_parent_class is not None:
            # cls_loss_qk_group = F.nll_loss(logits_qk_parent_class, y_group) # MNIST: logits.shape: (32, 2), y.shape: (32); CUB: logits.shape: (32, 200), y.shape: (32)
            cls_loss_qk_group, reg_loss_group = self.criterion(logits_qk_parent_class, F.one_hot(y_group, num_classes=5).float()) # for ANLS
        else:
            cls_loss_qk_group = 0
        # -----------------------------


        if stage == 'train':
            perf_dict = self.train_perf_calc.generate_dict_of_metrics(logits, y)
            group_perf_dict = self.train_group_perf_calc.generate_dict_of_metrics(logits_group, y_group)
            qk_group_perf_dict = self.train_qk_group_perf_calc.generate_dict_of_metrics(logits_qk_parent_class, y_group)
            # self.log()
        elif stage == 'val':
            perf_dict = self.val_perf_calc.generate_dict_of_metrics(logits, y)
            group_perf_dict = self.val_group_perf_calc.generate_dict_of_metrics(logits_group, y_group)
            qk_group_perf_dict = self.val_qk_group_perf_calc.generate_dict_of_metrics(logits_qk_parent_class, y_group)
            # self.log()
        elif stage == 'test':
            perf_dict = self.test_perf_calc.generate_dict_of_metrics(logits, y)
            group_perf_dict = self.test_group_perf_calc.generate_dict_of_metrics(logits_group, y_group)
            qk_group_perf_dict = self.test_qk_group_perf_calc.generate_dict_of_metrics(logits_qk_parent_class, y_group)

            # self.log()
            self.all_y.append(y)
            self.all_pred.append(preds)
            self.all_logit.append(logits)


        expl_loss = concepts_cost(concept_attn, expl) + spatial_concepts_cost( # PRED concept_attn.shape: (32, 1, 13), TARGET expl.shape: (32, 13); for glom crawl concept_attn.shape (990, 1, 630) and expl.shape (990, 630)
            spatial_concept_attn, spatial_expl # PRED spatial_concept_attn.shape: (32, 196, 95), TARGET spatial_expl.shpae: (32, 196, 95)
        )
        l1_loss = concepts_sparsity_cost(concept_attn, spatial_concept_attn)

        return (
            logits,
            preds,
            cls_loss,
            cls_loss_group,
            cls_loss_qk_group,
            perf_dict,
            group_perf_dict,
            qk_group_perf_dict,
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
                wandb.define_metric('val_acc', summary='max')
        # ==================== LaBo ends here ====================
        # ==================== ConceptTransformer starts here ====================
        '''
        TODO: Both `validation_step` have the same arguments `batch` and `batch_idx`. Both do not use batch_idx
        Need to check if the required `batch` is actually the argument `batch`
        '''
        logits, preds, cls_loss, cls_loss_group, cls_loss_qk_group, perf_dict, group_perf_dict, qk_group_perf_dict, expl_loss, l1_loss, _, _, _ = self.shared_step(batch, stage='val')
        self.log_dict(
            {
                "val_ce_loss": cls_loss,
                # "val_ce_loss_group": cls_loss_group,
                "val_qk_loss_group": cls_loss_qk_group,
                "val_acc": acc,
                # "val_acc_group": acc_group,
                "val_acc_qk_group": acc_qk_group,
                "val_expl_loss": expl_loss,
                "val_l1_loss": l1_loss,
            },
            on_step=False,
            on_epoch=True,
        )
        return acc
        # return ce_loss # LaBo
        # ==================== ConceptTransformer ends here ====================


    def test_step(self, batch, batch_idx):
        # ==================== ConceptTransformer starts here ====================
        '''
        TODO: do we need to append y and pred into overall lists?
        '''
        logits, preds, cls_loss, cls_loss_group, cls_loss_qk_group, perf_dict, group_perf_dict, qk_group_perf_dict, expl_loss, l1_loss, _, _, _ = self.shared_step(batch, stage='test')
        self.log_dict(
            {
                "test_ce_loss": cls_loss,
                # "test_ce_loss_group": cls_loss_group,
                "test_qk_loss_group": cls_loss_qk_group,
                "test_acc": acc,
                # "test_acc_group": acc_group,
                "test_acc_qk_group": acc_qk_group,
                "test_expl_loss": expl_loss,
                "test_l1_loss": l1_loss,
            }
        )
        return acc
        # return loss # LaBo
        # ==================== ConceptTransformer ends here ====================


    def test_epoch_end(self, outputs):
        # ==================== LaBo starts here ====================
        all_y = torch.hstack(self.all_y)
        all_pred = torch.hstack(self.all_pred)
        all_logit = torch.hstack(self.all_logit)
        self.total_test_acc = self.test_acc(all_logit, all_y)
        self.total_test_f1 = self.test_f1(all_pred, all_y)
        self.total_test_ap = self.test_ap(all_logit, all_y)
        pass
        # ==================== LaBo ends here ====================
    