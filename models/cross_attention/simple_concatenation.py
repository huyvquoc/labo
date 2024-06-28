import numpy as np
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import wandb

# from pytorch_lightning.pytorch.demos import Transformer
from loss.criterion import get_criterion


class LightningTransformer(pl.LightningModule):

    def __init__(self, cfg, init_weight=None, select_idx=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        concept_feat_path = self.data_root.joinpath('concepts_feat_{}.pth'.format(self.cfg.clip_model.replace('/','-')))
        concept_raw_path = self.data_root.joinpath('concepts_raw_selected.npy')
        concept2cls_path = self.data_root.joinpath('concept2cls_selected.npy')
        select_idx_path = self.data_root.joinpath('select_idx.pth')
        self.init_weight_save_dir = self.data_root.joinpath('init_weight.pth')
        cls_sim_path = self.data_root.joinpath('cls_sim.pth')

        if not concept_feat_path.exists():
            raise RuntimeError('need to call datamodule precompute_txt before using the model')
        else:
            if select_idx is None: self.select_idx = torch.load(select_idx_path)[:cfg.num_concept]
            else: self.select_idx = select_idx

            self.concepts = torch.load(concept_feat_path)[self.select_idx].cuda()
            if self.cfg.use_txt_norm: self.concepts = self.concepts / self.concepts.norm(dim=-1, keepdim=True)

            self.concept_raw = np.load(concept_raw_path)[self.select_idx]
            self.concept2cls = torch.from_numpy(np.load(concept2cls_path)[self.select_idx]).long().view(1, -1)

        if init_weight is None:
            self.init_weight_concept(self.concept2cls)
        else:
            self.init_weight = init_weight

        if 'cls_sim_prior' in self.cfg and self.cfg.cls_sim_prior and self.cfg.cls_sim_prior != 'none':
            # class similarity is prior to restrict class-concept association
            # if class A and B are dissimilar (similarity=0), then the mask location will be 0 
            print('use cls prior')
            cls_sim = torch.load(cls_sim_path)
            new_weights = []
            for concept_id in range(self.init_weight.shape[1]):
                target_class = int(torch.where(self.init_weight[:,concept_id] == 1)[0])
                new_weights.append(cls_sim[target_class] + self.init_weight[:,concept_id])
            self.init_weight = torch.vstack(new_weights).T
            # self.weight_mask = cls_sim @ self.init_weight

        self.asso_mat = torch.nn.Parameter(self.init_weight.clone())
        self.train_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task='multiclass')
        self.valid_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, average='micro', task='multiclass', top_k=cfg.top_k) # micro, macro, or weighted
        # self.test_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task='multiclass')
        self.test_f1 = torchmetrics.F1Score(num_classes=cfg.num_cls, average='macro', task='multiclass')
        self.test_ap = torchmetrics.AveragePrecision(num_classes=cfg.num_cls, average='macro', task='multiclass')

        self.all_y = []
        self.all_pred = []
        self.all_logit = []
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.cfg.num_cls)
        self.balanced_weights = torch.DoubleTensor(cfg.balanced_weights).cuda()
        self.criterion = get_criterion(cfg)
        self.criterion.cuda()
        self.save_hyperparameters()

    def forward(self, img_feat):
        self.concepts # (630, 512)
        img_feat # (1024, 512)
        return sim


    def training_step(self, train_batch, batch_idx):
        image, label = train_batch

        sim = self.forward(image)
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        # cls_loss = F.cross_entropy(pred, F.one_hot(label, num_classes=42).float(), weight=self.balanced_weights)
        cls_loss = F.binary_cross_entropy_with_logits(pred, F.one_hot(label, num_classes=42).float(), weight=self.balanced_weights)
        # cls_loss, _ = self.criterion(pred, F.one_hot(label, num_classes=42).float())
        if torch.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -torch.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = torch.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        else:
            # asso_mat sparse regulation
            row_l1_norm = torch.linalg.vector_norm(self.asso_mat, ord=1,
                                                dim=-1).max() #.mean()
        self.log('training_loss', cls_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        final_loss = cls_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=0.0000001)
        # opt = th.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return opt


    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        sim = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        self.log('val_loss', loss)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        image, y = batch
        sim = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.all_logit.append(pred)
        self.log('test_loss', loss)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.test_f1(pred, y)
        self.log('test_macro_f1', self.test_f1, on_step=False, on_epoch=True)
        self.test_ap(pred, y)
        self.log('test_ap', self.test_ap, on_step=False, on_epoch=True)
        return loss


    def test_epoch_end(self, outputs):
        all_y = torch.hstack(self.all_y)
        all_pred = torch.hstack(self.all_pred)
        all_logit = torch.hstack(self.all_logit)
        self.total_test_acc = self.test_acc(all_logit, all_y)
        self.total_test_f1 = self.test_f1(all_pred, all_y)
        self.total_test_ap = self.test_ap(all_logit, all_y)
        pass
    