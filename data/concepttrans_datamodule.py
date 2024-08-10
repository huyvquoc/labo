import os
import pickle

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
import torch
import torchvision 
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader

from .labo_datamodule import DataModule
from models.ctc.vit import extract_img_embs_using_conch_visual_trunk
import utils


def build_transform_dict(input_size=224, 
                         norm_mean=[0.485, 0.456, 0.406], 
                         norm_std=[0.229, 0.224, 0.225],
                         level=1):
    '''
    https://albumentations.ai/docs/examples/pytorch_classification/

    Inputs:
    - level(int): 1, 2, or 3. 
    1: fewest augmentations, 
    2: more augmentations,
    3: most augmentations, 
    '''
    if level == 1:
        # from ConceptTransformer code
        transform_dict = {'train': A.Compose([A.Resize(input_size, input_size),
                                              A.HorizontalFlip(p=0.5),
                                              A.Rotate(limit=(-30,30),p=1.0),
                                              A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                              ToTensorV2()]),
                          'val': A.Compose([A.Resize(input_size, input_size),
                                            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            ToTensorV2()]),
                          'test': A.Compose([A.Resize(input_size, input_size),
                                             A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                             ToTensorV2()])
        }

    elif level == 2:
        transform_dict = {
            'train': A.Compose([
                A.Flip(p=0.4), # 1
                A.Transpose(p=0.4),
                A.RandomRotate90(p=0.4),
                A.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE, p=0.4), # Rotate the input by an angle selected randomly from the uniform distribution
                A.Affine(scale=[0.65, 0.85], rotate=[-20,  20], shear=20, interpolation=3, fit_output=True, p=0.2),
                A.ShiftScaleRotate(p=0.2), # Randomly apply translate, scale and rotate the input
                # A.ElasticTransform(p=0.3),
                # A.GridDistortion(p=0.1),
                A.ColorJitter(p=0.15), # 2 color
                A.HueSaturationValue(p=0.15), # Randomly change hue, saturation and value of the inpunt image
                A.RGBShift(p=0.15), # Randomly shift values for each channel of the input RGB image
                A.RandomGamma(p=0.15),
                # A.GaussianBlur(p=0.3),
                # A.GlassBlur(p=0.3),
                A.Sharpen(p=0.2),
                A.Downscale(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                # A.CLAHE(p=0.2),
                A.Equalize(p=0.2),
                # A.RandomToneCurve(p=0.1),
                # A.GaussNoise(p=0.1), # Apply gausssssian noise to the input image. Don't see clear
                A.Resize(height=input_size, width=input_size, p=1),
                A.Normalize(mean=norm_mean, std=norm_std, p=1),
                ToTensorV2()
            ]),
            'val': A.Compose([
                A.Resize(height=input_size, width=input_size, p=1),
                # transforms.Resize(int(input_size/7*8)),
                # transforms.CenterCrop(input_size),
                A.Normalize(mean=norm_mean, std=norm_std, p=1),
                ToTensorV2()
            ]),
            'test': A.Compose([
                A.Resize(height=input_size, width=input_size, p=1),
                # transforms.Resize(int(input_size/7*8)),
                # transforms.CenterCrop(input_size),
                A.Normalize(mean=norm_mean, std=norm_std, p=1),
                ToTensorV2()
            ]),
        }

    elif level == 3:
        transform_dict = {
            'train': A.Compose([
                A.Flip(p=0.4), # 1
                A.Transpose(p=0.4),
                A.RandomRotate90(p=0.4),
                A.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE, p=0.4), # Rotate the input by an angle selected randomly from the uniform distribution
                A.Affine(scale=[0.65, 0.85], rotate=[-20,  20], shear=20, interpolation=3, fit_output=True, p=0.8),
                A.ShiftScaleRotate(p=0.2), # Randomly apply translate, scale and rotate the input
                # A.ElasticTransform(p=0.3),
                # A.GridDistortion(p=0.1),
                A.ColorJitter(p=0.3), # 2 color
                A.HueSaturationValue(p=0.3), # Randomly change hue, saturation and value of the inpunt image
                A.RGBShift(p=0.2), # Randomly shift values for each channel of the input RGB image
                A.RandomGamma(p=0.2),
                A.GaussianBlur(p=0.3),
                # A.GlassBlur(p=0.3),
                A.Sharpen(p=0.3),
                A.Downscale(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.CLAHE(p=0.2),
                # A.Equalize(p=0.2),
                # A.RandomToneCurve(p=0.1),
                A.GaussNoise(p=0.1), # Apply gausssssian noise to the input image. Don't see clear
                A.Resize(height=input_size, width=input_size, p=1),
                A.Normalize(mean=norm_mean, std=norm_std, p=1),
                ToTensorV2()
            ]),
            'val': A.Compose([
                A.Resize(height=input_size, width=input_size, p=1),
                # transforms.Resize(int(input_size/7*8)),
                # transforms.CenterCrop(input_size),
                A.Normalize(mean=norm_mean, std=norm_std, p=1),
                ToTensorV2()
            ]),
            'test': A.Compose([
                A.Resize(height=input_size, width=input_size, p=1),
                # transforms.Resize(int(input_size/7*8)),
                # transforms.CenterCrop(input_size),
                A.Normalize(mean=norm_mean, std=norm_std, p=1),
                ToTensorV2()
            ]),
        }

    return transform_dict

class ImgPreExtEmbsDataset(Dataset):
    '''
    Only for pre extraction of image embeddings
    '''
    def __init__(self, img_filepaths, labels, transform=None):
        super().__init__()

        self.loader = default_loader

        self.img_filepaths = img_filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_filepaths)

    def __getitem__(self, idx):
        img_filepath = self.img_filepaths[idx]
        img = self.loader(img_filepath)

        if self.transform is not None: # self.transform is passed through the datamodule in the file 'cub2011parts_datamodule'
            img_np = np.array(img) # img is a PIL with (153, 264); image_np.shape: (264, 153, 3)
            img = self.transform(image=img_np)["image"]

        label = self.labels[idx]

        return img, label

class ImgExplDataset(Dataset):
    def __init__(self, \
                 model_name=None, \
                 list_of_img_names=[], \
                 list_of_labels=[], \
                 cls_names=None, \
                 groupname2clsname=None,
                 num_concept=-1,
                 concept2cls=None, 
                 split_name='train',
                 img_embeddings_save_dir=None,
                 labels_save_dir=None,
                 use_extracted_embs=True,
                 only_visual_trunk=False,
                 do_aug=False,
                 n_iters=10,
                 transform=None):
        super().__init__()

        self.loader = default_loader

        self.model_name = model_name
        self.img_names = list_of_img_names # train, val, or test. read pickle file in pl.DataModule
        self.labels = list_of_labels
        self.num_concept = num_concept
        self.concept2cls = concept2cls.tolist()[0]
        self.do_aug = do_aug
        self.n_iters = n_iters
        self.transform = transform

        # Mapping class label to selected concepts'new indices. These new indices are not the indices in `self.select_idx`, but are the new indices have been assigned to the selected indices (0 -> 629)
        self.cls_names = cls_names
        self.cls2indices = {}
        for idx, cls in enumerate(self.concept2cls):
            if cls not in self.cls2indices:
                self.cls2indices[cls] = []
            self.cls2indices[cls].append(idx)

        self.groupname2clsname = groupname2clsname
        self.groupnames = list(groupname2clsname.keys())
        self.clsname2groupname = {}
        for grp_name, cls_names in self.groupname2clsname.items():
            for cls_name in cls_names:
                self.clsname2groupname[cls_name] = grp_name

        # ----- pre-extraction of image embeddings -----
        if use_extracted_embs:
            print('- Use img_embeddings_save_dir: ', img_embeddings_save_dir)
            if Path(img_embeddings_save_dir.format(split_name)).exists():
                print('Exist! Just load to use')
                self.img_embs = torch.load(img_embeddings_save_dir.format(split_name))
                self.labels = torch.load(labels_save_dir.format(split_name))
            else:
                print('Not exist! Jump into a function to start the extraction')
                if split_name == 'train':
                    self.img_embs, self.labels = extract_img_embs_using_conch_visual_trunk(model_name=self.model_name, list_of_img_paths=self.img_names, list_of_labels=self.labels, dataset_class=ImgPreExtEmbsDataset, \
                                                                                           transform_compose=self.transform, only_visual_trunk=only_visual_trunk, do_aug=self.do_aug, n_iters=self.n_iters)
                else:
                    self.img_embs, self.labels = extract_img_embs_using_conch_visual_trunk(model_name=self.model_name, list_of_img_paths=self.img_names, list_of_labels=self.labels, dataset_class=ImgPreExtEmbsDataset, \
                                                                                           transform_compose=self.transform, only_visual_trunk=only_visual_trunk, do_aug=False, n_iters=self.n_iters)
                torch.save(self.img_embs, img_embeddings_save_dir.format(split_name))
                torch.save(self.labels, labels_save_dir.format(split_name))
        
        self.use_extracted_embs = use_extracted_embs

    def __len__(self):
        if self.use_extracted_embs:
            return len(self.img_embs)
        else:
            return len(self.img_names)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (PIL image, target) where target is index of the target class.
            or:
            tupe: (image embedding, target)
        """
        int_target = self.labels[idx]
        target = torch.tensor(int_target, dtype=torch.long)

        str_target = self.cls_names[int_target]
        
        int_target_group = self.groupnames.index(self.clsname2groupname[str_target]) # inferred from target int
        target_group = torch.tensor(int_target_group, dtype=torch.long)

        if self.use_extracted_embs:
            img = self.img_embs[idx]
        else:
            img_path = self.img_names[idx] # Absolute path to a specific image
            img = self.loader(img_path) # PIL image object

            if self.transform is not None: # self.transform is passed through the datamodule in the file 'cub2011parts_datamodule'
                image_np = np.array(img) # img is a PIL with (153, 264); image_np.shape: (264, 153, 3)
                transformed = self.transform(image=image_np) # a dictionary of a key 'image'
                img = transformed['image'] # shape: (3, 448, 448)

        expl = self._get_explanations(int_target)

        # spatial concepts are always set to None
        return (img, expl, torch.tensor(float('nan')), target, target_group)
    
    def _get_explanations(self, cls):
        '''
        target: is label, int or str?
        '''
        # Global explanations
        expl = torch.FloatTensor(self.num_concept).fill_(0) # self.num_non_spatial_attributes = 13, initialize all elements with nan, expl.shape = (13), for glom crawl: expl.shape = (630)

        for index in self.cls2indices[cls]: # map each class to all of its concept indices
            expl[index] = 1

        return expl # target expl and target spatial_expl is just to see which attributes exist in the current patch, just 0 or 1 value
    
class ImgExplDataModule(pl.LightningDataModule):

    def __init__(self, cfg, data_root, img_root, img_split_path, cls_name_path, img_ext, num_concept, batch_size, num_workers, on_gpu=False, model_name='conch_ViT-B-16', use_txt_norm=False):
        super().__init__()

        self.cfg = cfg
        self.data_root = Path(data_root)
        self.img_root = Path(img_root)
        self.img_split_path = Path(img_split_path)
        self.img_ext = img_ext
        self.num_concept = num_concept
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.on_gpu = on_gpu
        self.use_txt_norm = use_txt_norm

        self.cls_names = np.load(cls_name_path).tolist() # for reference, the mapping between indices and names
        self.cls_names.sort()

        self.splits = {
            split: utils.pickle_load(
                self.img_split_path.joinpath(
                    'class2images_{}.p'.format(split)))
            for split in ['train', 'val', 'test']
        }

        self.dict_of_labels = {'train': [], 'val': [], 'test': []}
        self.dict_of_img_paths = {'train': [], 'val': [], 'test': []}
        for mode in ['train', 'val', 'test']:
            cls2img = self.splits[mode]
            for cls_name, img_names in cls2img.items():
                self.dict_of_labels[mode].extend([self.cls_names.index(cls_name)] * len(img_names))
                self.dict_of_img_paths[mode].extend([self.img_root.joinpath('{}{}'.format(img_name, self.img_ext)) for img_name in img_names])

        # self.concept_feat_path = self.data_root.joinpath('concepts_feat_{}.pth'.format(model_name))
        # self.concept_raw_path = self.data_root.joinpath('concepts_raw_selected.npy')
        self.concept2cls_path = self.data_root.joinpath('concept2cls_selected.npy') # BEING USED
        self.select_idx_path = self.data_root.joinpath('select_idx.pth') # BEING USED
            
        self.select_idx = torch.load(self.select_idx_path)[:self.num_concept] # (630, ) # BEING USED
        # self.concepts = torch.load(self.concept_feat_path)[self.select_idx].cuda() # shape: (630, 512)
        # if self.use_txt_norm: self.concepts = self.concepts / self.concepts.norm(dim=-1, keepdim=True)
        # self.concept_raw = np.load(self.concept_raw_path)[self.select_idx] # shape: (630, )
        self.concept2cls = torch.from_numpy(np.load(self.concept2cls_path)[self.select_idx]).long().view(1, -1) # shape: (1, 630) # BEING USED

        if model_name == 'conch_ViT-B-16':
            resized_img_dim = 448
            mean_tuple = (0.48145466, 0.4578275, 0.40821073)
            std_tuple = (0.26862954, 0.26130258, 0.27577711)
        else:
            # === OPENCLIP ===
            resized_img_dim = 224
            mean_tuple = (0.5, 0.5, 0.5)
            std_tuple = (0.5, 0.5, 0.5)

        self.transform = build_transform_dict(input_size=resized_img_dim, 
                                              norm_mean=mean_tuple, # these numbers are taken from CONCH's preprocess
                                              norm_std=std_tuple, # these numbers are taken from CONCH's preprocess
                                              level=self.cfg.aug_level)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        '''
        A correct way to transfer batch to device when you return list, tuple, dict, ... in Dataset `getitem`

        Ref:
        [1] https://github.com/Lightning-AI/pytorch-lightning/discussions/14616#discussioncomment-3605304
        [2] https://lightning.ai/docs/pytorch/latest/data/datamodule.html#transfer-batch-to-device
        [3] https://github.com/Lightning-AI/pytorch-lightning/issues/18124
        '''
        if isinstance(batch, list):
            batch[0] = batch[0].to(device)
            batch[1] = batch[1].to(device)
            batch[2] = batch[2].to(device)
            batch[3] = batch[3].to(device)
            batch[4] = batch[4].to(device)
            return batch
        else:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        
    def get_cls_names(self):
        return self.cls_names
 
    def setup(self, stage=None):
        """
        Set up datasets for dataloader to load from. Depending on the need, return either:
        - (img_feat, label), concept_feat will be loaded in the model
        - (the dot product between img_feat and concept_feat, label)
        - if allowing grad to image, provide (image, label)
        - if allowing grad to text, compute concept_feat inside the model        
        """
        # check if val and test set are completely different
        for img_val_path in self.dict_of_img_paths['val']:
            assert img_val_path not in self.dict_of_img_paths['test']

        self.datasets = {
            mode: ImgExplDataset(model_name=self.cfg.clip_model,
                                 list_of_img_names=self.dict_of_img_paths[mode], \
                                 list_of_labels=self.dict_of_labels[mode], \
                                 cls_names=self.cls_names, \
                                 groupname2clsname=self.cfg.groupname2clsname, \
                                 num_concept=self.num_concept, \
                                 concept2cls=self.concept2cls, \
                                 split_name=mode, \
                                 img_embeddings_save_dir=self.cfg.img_embeddings_save_dir,
                                 labels_save_dir=self.cfg.labels_save_dir,
                                 use_extracted_embs=self.cfg.use_extracted_img_embs,
                                 only_visual_trunk=self.cfg.only_visual_trunk,
                                 do_aug=self.cfg.do_aug,
                                 n_iters=self.cfg.n_iters,
                                 transform=self.transform[mode])
            for mode in ['train', 'val', 'test']
        }
    
    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def val_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def predict_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False) 