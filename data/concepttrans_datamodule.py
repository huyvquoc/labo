import os
import pickle

import albumentations as A
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader


class ImgExplDataset(VisionDataset):
    def __init__(self, img_path='/data/hqvo3/Datasets/vlm/ml-scripts/datasets/glom_refine_labo', \
                 data_root='exp/asso_opt/GlomCrawl/GlomCrawl_allshot_fac',
                 img_names=[], labels=[], model_name='conch_ViT-B-16', num_concepts=None, train=True,
             transform=None):
        super().__init__(img_path, transform=transform)

        self.loader = default_loader
        self.num_concepts = num_concepts # read from cfg.num_concepts
        self.train = train

        self.img_path = Path(self.img_path)
        self.data_root = Path(data_root)
        self.img_names = img_names # train, val, or test. read pickle file in pl.DataModule
        self.labels = labels
        self.concept_feat_path = self.data_root.joinpath('concepts_feat_{}.pth'.format(model_name))
        self.concept_raw_path = self.data_root.joinpath('concepts_raw_selected.npy')
        self.concept2cls_path = self.data_root.joinpath('concept2cls_selected.npy')
        self.select_idx_path = self.data_root.joinpath('select_idx.pth')
        # TODO: create a dictionary of mapping from cls to idx

        self.concepts = torch.load(self.concept_feat_path)[self.select_idx].cuda() # shape: (630, 512)
        if self.cfg.use_txt_norm: self.concepts = self.concepts / self.concepts.norm(dim=-1, keepdim=True)
        self.concept_raw = np.load(self.concept_raw_path)[self.select_idx] # shape: (630, )
        self.concept2cls = torch.from_numpy(np.load(self.concept2cls_path)[self.select_idx]).long().view(1, -1) # shape: (1, 630)
        self.select_idx = torch.load(self.select_idx_path)[num_concepts] # (630, )

        # Mapping class label to selected concepts'new indices. These new indices are not the indices in `self.select_idx`, but are the new indices have been assigned to the selected indices (0 -> 629)
        self.cls2indices = {}
        for idx, (_, cls) in enumerate(self.concept2cls.items()):
            if cls not in self.cls2indices:
                self.cls2indices[cls] = []
            self.cls2indices[cls].append(idx)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (PIL image, target) where target is index of the target class.
        """
        img_path = self.img_path.joinpath(self.img_names[idx]) # Absolute path to a specific image
        target = self.labels[idx]

        img = self.loader(img_path)

        if self.transform is not None: # self.transform is passed through the datamodule in the file 'cub2011parts_datamodule'
            image_np = np.array(img) # img is a PIL with (153, 264); image_np.shape: (264, 153, 3)
            transformed = self.transform(image=image_np)
            img = transformed['image'] # shape: (3, 224, 224)

        expl = self._get_explanations(target)

        return (img, expl, None, target)
    
    def _get_explanations(self, cls):
        '''
        target: is label, int or str?
        '''
        # Global explanations
        expl = torch.FloatTensor(self.num_concepts).fill_(float('nan')) # self.num_non_spatial_attributes = 13, initialize all elements with nan, expl.shape = (13)

        for index in self.cls2indices[cls]:
            expl[index] = 1

        return expl # target expl and target spatial_expl is just to see which attributes exist in the current patch, just 0 or 1 value
    
class ImgExplDataModule(pl.LightningDataModule):
    def __init__(self, img_path, data_root, batch_size, num_workers):
        super().__init__()

        self.img_path = img_path
        self.data_root = data_root

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = A.Compose([])

        self.test_transform = A.Compose([])
    
    def prepare_data(self):
        ImgExplDataset(img_path, data_root, \
                       img_names=, \
                       labels=, \
                       model_name=, \
                       num_concepts=, \
                       train=, \
                       transform=)

    def setup(self, stage):
        """
        Set up datasets for dataloader to load from. Depending on the need, return either:
        - (img_feat, label), concept_feat will be loaded in the model
        - (the dot product between img_feat and concept_feat, label)
        - if allowing grad to image, provide (image, label)
        - if allowing grad to text, compute concept_feat inside the model        
        """
        self.datasets = {
            mode: ImgExplDataset(img_path=, \
                                 data_root=, \
                                 self.label[mode],
                                   self.on_gpu)
            for mode in ['train', 'val', 'test']
        }

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def val_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            batch_size=self.bs,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            batch_size=self.bs,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def predict_dataloader(self):
        test_img_paths = self.get_img_n_shot(self.splits['test'], 'all')[0]
        return DataLoader(
            Dataset_with_name(self.datasets['test'], test_img_paths),
            batch_size=self.bs,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)