import collections
import math

from pathlib import Path
from PIL import Image
import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm

from .ctc import CrossAttention
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
# from data.pre_ext_dataset import ImgPreExtEmbsDataset
from models.ctc.prompts import HierarchyAwarePromptLearner
import open_clip


def cub_cvit(cfg, class_names, concept2cls, baseline=False, *args, **kwargs):
    """
    Args:
        baseline (bool): If true it returns the baseline model, which in this case it just the vit backbone without concept transformer
    """
    if not baseline:
        return CVIT(
            cfg,
            model_name=cfg.clip_model,
            num_classes=cfg.num_cls,
            class_names=class_names,
            concept2cls=concept2cls,
            n_unsup_concepts=cfg.num_unsup_concept,
            n_concepts=cfg.num_concept,
            n_spatial_concepts=cfg.num_spatial_concept,
            num_heads=cfg.num_heads,
            attention_dropout=cfg.attention_dropout,
            projection_dropout=cfg.projection_dropout,
            ckpt_path=cfg.ckpt_path,
            concept_embeddings_save_dir=cfg.concept_embeddings_save_dir,
            use_extracted_img_embs=cfg.use_extracted_img_embs,
            use_patches_embeddings=cfg.use_patches_embeddings,
            only_visual_trunk=cfg.only_visual_trunk,
            set_learnable_prompts=cfg.set_learnable_prompts,
            *args,
            **kwargs,
        )
    else:
        return ExplVIT(model_name=cfg.clip_model, num_classes=200)


class ExplVIT(VisionTransformer):
    """VIT modified to return dummy concept attentions"""
    def __init__(self, num_classes=200, model_name="vit_base_patch16_224", *args, **kwargs):
        super().__init__(num_classes=num_classes)

        loaded_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.load_state_dict(loaded_model.state_dict())

    def forward(self, x):
        out = super().forward(x)
        return out, None, None, None

def one_forward_of_vlm_foundation_img_branch(feature_extractor, preprocessed_img, only_visual_trunk=False):
    if only_visual_trunk:
        with torch.inference_mode():
            x = feature_extractor.patch_embed(preprocessed_img) # (32, 196, 1024), TODO: CoCa model.visual.trunk.patch_embed: Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16)) shape (128, 14, 14, 768)
            x = torch.flatten(x, start_dim=1, end_dim=2) # (128, 196, 768)

            cls_token = feature_extractor.cls_token.expand( # self.feature_extractor.cls_token.shape: (1, 1, 1024)
                x.shape[0], -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks, cls_token.shape: (32, 1, 1024), TODO: CoCa model.visual.trunk.cls_token (torch.nn.parameter.Parameter) with shape (990, 1, 768)

            # self.feature_extractor.dist_token.shape: (1, 1, 1024)
            if not hasattr(feature_extractor, 'dist_token') or feature_extractor.dist_token is None: # yes, dist token is None, TODO: CoCa has no dist_token too: model.visual.trunk.dist_token
                x = torch.cat((cls_token, x), dim=1) # x.shape: (32, 197, 1024)
            else:
                x = torch.cat((cls_token, feature_extractor.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
                
            # TODO: CoCa model.visual.trunk.pos_embed.shape: (1, 785, 768), model.visual.trunk.pos_drop: Dropout(p=0.0, inplace=False)
            x = feature_extractor.pos_drop(x + feature_extractor.pos_embed) # self.feature_extractor.pos_embed.shape: (1, 197, 1024), x.shape: (32, 197, 1024)
            # TODO: CoCa model.visual.trunk.blocks: 12 blocks (id 0 -> 11)
            x = feature_extractor.blocks(x) # shape: (32, 197, 1024)
            # TODO: CoCa model.visual.trunk.norm: LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            x = feature_extractor.norm(x) # shape: (32, 197, 1024)
            # list_of_img_embs.append(x)
    else:
        with torch.inference_mode():
            x = feature_extractor.encode_image(preprocessed_img)
            # list_of_img_embs.append(x)

    return x

def extract_img_embs_using_conch_visual_trunk(model_name, list_of_img_paths, list_of_labels, dataset_class, transform_compose, only_visual_trunk=False, ckpt_path='/data/hqvo3/weights/CONCH/pytorch_model.bin', do_aug=False, n_iters=10):
    '''
    This function is mostly used in the concepttrans data module file

    Inputs:
    - list_of_img_paths (list): list of absolute image paths
    - do_aug (bool): whether using augmentation or not
    - n_iters (int): if do augmentation, how many iters to simulate epochs? For each original sample, we will have n_iters augmented versions of it
    '''
    if do_aug:
        predict_dataset = dataset_class(img_filepaths=list_of_img_paths, labels=list_of_labels, transform=transform_compose)
        predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) # TODO: change batch size

    print('- Start extract img embeddings...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'conch_ViT-B-16':
        feature_extractor, preprocess = create_model_from_pretrained(model_name, ckpt_path, device=device)
        if only_visual_trunk:
            feature_extractor = feature_extractor.visual.trunk
    else:
        # === OPENCLIP ===
        feature_extractor, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s32b_b82k', device=device)


    list_of_img_embs = []
    list_of_new_labels = [] # only change when there are augmentations
    if do_aug:
      for idx in range(n_iters):
          print('[**] aug idx: ', idx)
          stream = tqdm.tqdm(predict_loader)
          for jdx, (images, labels) in enumerate(stream, start=1):
              images = images.to(device, non_blocking=True)
              # labels = labels.to(device, non_blocking=True)
              img_embs = one_forward_of_vlm_foundation_img_branch(feature_extractor, images, only_visual_trunk)
              list_of_img_embs.append(img_embs.to("cpu")) # TODO: when batch size is greater than 1, do we need to change from append to extend
              list_of_new_labels.append(labels.item())
    else:
      for img_path in list_of_img_paths:
          preprocessed_img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
          list_of_img_embs.append(one_forward_of_vlm_foundation_img_branch(feature_extractor, preprocessed_img, only_visual_trunk))
      list_of_new_labels = list_of_labels.copy()

    print('- Complete extract img embeddings!')
    
    ret = torch.cat(list_of_img_embs)
    if ret.ndim == 2:
        ret = ret[:, None, :]
    return ret, list_of_new_labels

# def conch_coca_text_tower():

class CVIT(nn.Module):
    """Concept Vision Transformer"""
    def __init__(
        self,
        cfg,
        num_classes=200,
        class_names=None,
        concept2cls=None,
        model_name="vit_base_patch16_224",
        pretrained=True,
        n_unsup_concepts=50,
        n_concepts=13,
        n_spatial_concepts=95,
        num_heads=12,
        attention_dropout=0.1,
        projection_dropout=0.1,
        ckpt_path=None, # only for loading conch model
        concept_embeddings_save_dir=None,
        use_extracted_img_embs=True,
        use_patches_embeddings=False,
        only_visual_trunk=False,
        set_learnable_prompts=False,
        *args,
        **kwargs,
    ):
        '''
        Inputs:
        - only_visual_trunk: whether to only use embedding from visual trunk of CONCH, or use the one from attn_contrast
        '''
        super().__init__()

        self.model_name = model_name

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model_name == 'vit_base_patch16_224':
            self.feature_extractor = timm.create_model(self.model_name, pretrained=True, num_classes=num_classes) # CoCa model.visual
            del self.feature_extractor.head
        elif self.model_name == 'conch_ViT-B-16':
            self.feature_extractor, self.preprocess = create_model_from_pretrained(self.model_name, ckpt_path, device=device)
            if only_visual_trunk:
                self.feature_extractor = self.feature_extractor.visual.trunk
            # _ = self.feature_extractor.eval()
            self.embed_dim = self.feature_extractor.embed_dim
        else:
            # === OPENCLIP ===
            # model, preprocess_train, preprocess_val
            self.feature_extractor, self.preprocess, _ = open_clip.create_model_and_transforms(self.model_name, pretrained='laion2b_s32b_b82k', device=device)
            self.embed_dim = self.feature_extractor.ln_final.weight.shape[0]

        self.set_learnable_prompts = set_learnable_prompts
        self.concept2cls = torch.squeeze(concept2cls)

        self.classifier = ConceptTransformerVIT(
            cfg,
            embedding_dim=self.embed_dim, # 1024
            num_classes=num_classes, # 200
            class_names=class_names,
            num_heads=num_heads,
            attention_dropout=attention_dropout, # 0.1
            projection_dropout=projection_dropout, # 0.1
            n_unsup_concepts=n_unsup_concepts, # 0
            n_concepts=n_concepts, # 13
            n_spatial_concepts=n_spatial_concepts, # 95
            model_name=self.model_name,
            ckpt_path=ckpt_path,
            concept_embeddings_save_dir=concept_embeddings_save_dir,
            only_visual_trunk=only_visual_trunk,
            set_learnable_prompts=self.set_learnable_prompts,
            concept2cls=self.concept2cls,
            *args,
            **kwargs, # {'batch_size': 32, 'debug': False, 'data_dir': '/data/hqvo3/data/cub2011', 'finetune_unfreeze_epoch': 0, 'num_workers': 8, 'seed': 42, 'data_name': 'CUB2011Parts'}
        )

        self.use_extracted_img_embs = use_extracted_img_embs
        self.use_patches_embeddings = use_patches_embeddings

    def forward(self, x, x_concepts):
        '''
        Inputs:
        - Original images - x.shape: (batch_size, 3, 224, 224) or (batch_size, 3, 448, 448) for GlomCrawl
        - x_concepts: can be original raw concepts `self.concept_raw` (630,) OR extracted embeddings `self.concepts` (630, 512)
        OR
        - Extracted embeds - x.shape: (batch_size, 512) or (batch_size, 768) for conch.vision.trunk
        '''
        if not self.use_extracted_img_embs:
            if self.use_patches_embeddings:
                x = self.feature_extractor.patch_embed(x) # (32, 196, 1024), TODO: CoCa model.visual.trunk.patch_embed: Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16)) shape (128, 14, 14, 768)
                if self.model_name == 'conch_ViT-B-16':
                    x = torch.flatten(x, start_dim=1, end_dim=2) # (128, 196, 768)

                cls_token = self.feature_extractor.cls_token.expand( # self.feature_extractor.cls_token.shape: (1, 1, 1024)
                    x.shape[0], -1, -1
                )  # stole cls_tokens impl from Phil Wang, thanks, cls_token.shape: (32, 1, 1024), TODO: CoCa model.visual.trunk.cls_token (torch.nn.parameter.Parameter) with shape (990, 1, 768)

                # self.feature_extractor.dist_token.shape: (1, 1, 1024)
                if not hasattr(self.feature_extractor, 'dist_token') or self.feature_extractor.dist_token is None: # yes, dist token is None, TODO: CoCa has no dist_token too: model.visual.trunk.dist_token
                    x = torch.cat((cls_token, x), dim=1) # x.shape: (32, 197, 1024)
                else:
                    x = torch.cat((cls_token, self.feature_extractor.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        
                # TODO: CoCa model.visual.trunk.pos_embed.shape: (1, 785, 768), model.visual.trunk.pos_drop: Dropout(p=0.0, inplace=False)
                x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed) # self.feature_extractor.pos_embed.shape: (1, 197, 1024), x.shape: (32, 197, 1024)
                # TODO: CoCa model.visual.trunk.blocks: 12 blocks (id 0 -> 11)
                x = self.feature_extractor.blocks(x) # shape: (32, 197, 1024)
                # TODO: CoCa model.visual.trunk.norm: LayerNorm((768,), eps=1e-06, elementwise_affine=True)
                x = self.feature_extractor.norm(x) # shape: (32, 197, 1024)

                x /= x.norm(dim=-1, keepdim=True) # shape (990, 1, 512)
            else:
                # === for training with unfrozen vision layers ===
                # Reference from CONCH's authors: /conch/open_clip_custom/vision_tower.py 
                # x is an image tensor with shape(batch_size, 3, 448, 448)
                # image_latent, tokens_embs = self.feature_extractor.visual(x)

                x = self.feature_extractor.visual.trunk(x) # (990, 785, 768)
                x = self.feature_extractor.visual.attn_pool_contrast(x)[:, 0] # single query, shape (990, 512)
                x = self.feature_extractor.visual.ln_contrast(x) # shape (990, 512)
                x = x @ self.feature_extractor.visual.proj_contrast # shape (990, 512)

                x = F.normalize(x, dim=-1) # shape (990, 512)
                x = x[:, None, :]
 

        out, out_group, unsup_concept_attn, concept_attn, spatial_concept_attn, qk_parent_class = self.classifier(
            x[:, 0].unsqueeze(1), x[:, 1:], x_concepts # shape: cls token: (32, 1, 1024) and remaining: (32, 196, 1024)
        )
        # out.shape: (32, 200)
        # unsup_concept_attn: None
        # concept_attn.shape: (32, 1, 13)
        # spatial_concept_attn.shape: (32, 196, 95)
        return out, out_group, unsup_concept_attn, concept_attn, spatial_concept_attn, qk_parent_class


class ConceptTransformerVIT(nn.Module):
    """Processes spatial and non-spatial concepts in parallel and aggregates the log-probabilities at the end.
    The difference with the version in ctc.py is that instead of using sequence pooling for global concepts it
    uses the embedding of the cls token of the VIT

    3 cases:
    (1) soft concepts by default of ConceptTransformer
    (2) hard concepts in raw
    (3) hard concepts in the form of extracted embeddings
    (4) soft concepts feeding into CONCH
    """
    def __init__(
        self,
        cfg,
        embedding_dim=768,
        num_classes=10,
        class_names=None,
        num_heads=2,
        attention_dropout=0.1,
        projection_dropout=0.1,
        n_unsup_concepts=10,
        n_concepts=10,
        n_spatial_concepts=10,
        model_name=None,
        ckpt_path=None,
        concept_embeddings_save_dir=None,
        only_visual_trunk=False,
        set_learnable_prompts=False,
        concept2cls=None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model_name = model_name
        self.only_visual_trunk = only_visual_trunk
        self.class_names = class_names
        self.concept2cls = concept2cls

        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.model_name == 'conch_ViT-B-16':
            # ----- only for CONCH -----
            self.tokenizer = get_tokenizer()
            self.model, self.preprocess = create_model_from_pretrained(self.model_name, ckpt_path, device=self.device)
            # _ = self.model.eval()
            # ----- only for CONCH -----

            self.concept_embeddings_save_dir = Path(concept_embeddings_save_dir)
            self.concept_embeddings = None
            if self.concept_embeddings_save_dir.exists():
                self.concept_embeddings = torch.load(self.concept_embeddings_save_dir).cuda()
        else:
            # === OPENCLIP ===
            self.tokenizer = open_clip.get_tokenizer(self.model_name) # 'ViT-L-14'
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained='laion2b_s32b_b82k', device=self.device) # 'ViT-L-14'

            self.concept_embeddings_save_dir = Path(concept_embeddings_save_dir)
            self.concept_embeddings = None
            if self.concept_embeddings_save_dir.exists():
                self.concept_embeddings = torch.load(self.concept_embeddings_save_dir).cuda()

        # Unsupervised concepts
        self.n_unsup_concepts = n_unsup_concepts
        self.unsup_concepts = nn.Parameter(
            torch.zeros(1, n_unsup_concepts, embedding_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.unsup_concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_unsup_concepts > 0:
            self.unsup_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Non-spatial concepts
        self.n_concepts = n_concepts
        # TODO: remove the next 2 lines, initialized with text embedding from CONCH text encoder shape (, 512)
        # TODO: Do we still need nn.Parameter? Is nn.Parameter for soft prompting?
        self.concepts = nn.Parameter(torch.zeros(1, n_concepts, embedding_dim), requires_grad=True) # shape: (1, 13, 1024)
        nn.init.trunc_normal_(self.concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_concepts > 0:
            self.set_learnable_prompts = set_learnable_prompts

            n_group_outputs = None
            if self.set_learnable_prompts:
                self.prompt_learner = HierarchyAwarePromptLearner(cfg, classnames=self.class_names, groupname2clsname=cfg.groupname2clsname, 
                                                                  clip_model=self.model, tokenizer=self.tokenizer, 
                                                                  use_group=True, multilabel_on=False)
                self.groupnames, self.groupname2clsname = self.prompt_learner.get_groupname2clsname()
                self.clsname2groupname = {}
                for grp_name, cls_names in self.groupname2clsname.items():
                    for cls_name in cls_names:
                        self.clsname2groupname[cls_name] = grp_name

                n_group_outputs = len(self.groupnames)

                dict_of_mapping_from_conceptid_to_softpromptid = {} # concept index (630 concepts) to parent class index (5 soft prompts for)
                for concept_idx in range(self.concept2cls.shape[0]): # 630
                    corr_label_int = int(self.concept2cls[concept_idx]) # corresponding child class in int
                    corr_label_str = self.class_names[corr_label_int] # corresponding child class in str
                    corr_parent_label_str = self.clsname2groupname[corr_label_str] # corresponding parent class in str
                    corr_parent_label_int = self.groupnames.index(corr_parent_label_str) # corresponding parent class in int

                    dict_of_mapping_from_conceptid_to_softpromptid[concept_idx] = corr_parent_label_int

                self.dict_of_mapping_from_conceptid_to_softpromptid = collections.OrderedDict(sorted(dict_of_mapping_from_conceptid_to_softpromptid.items()))
            
            self.concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                n_group_outputs=n_group_outputs,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
                dict_of_mapping_from_conceptid_to_softpromptid=self.dict_of_mapping_from_conceptid_to_softpromptid,
                use_child_classes=cfg.use_child_classes,
                use_parent_classes=cfg.use_parent_classes,
                use_attn_parent_branch=cfg.use_attn_parent_branch
            )

        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        # TODO: remove the next 2 lines, initialized with text embedding from CONCH text encoder shape (, 512)
        # TODO: Do we still need nn.Parameter? Is nn.Parameter for soft prompting?
        self.spatial_concepts = nn.Parameter(
            torch.zeros(1, n_spatial_concepts, embedding_dim), requires_grad=True # shape: (1, 95, 1024)
        )
        nn.init.trunc_normal_(self.spatial_concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_spatial_concepts > 0:
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def _encode_learnable_prompts(self, text_after_token_emb=None, text_after_tokenize=None, normalize=True, embed_cls=False):
        '''
        Copy from `conch/open_clip_custom/coca_model.py`

        **For learnable soft prompts**

        By default, do normalize (True) and embed_cls (True)

        Input:
        - text_after_token_emb: (630, 127, 768)
        - text_after_tokenize: is text tokens (630, 128)
        '''
        text_after_tokenize = text_after_tokenize[:, :-1] if embed_cls else text_after_tokenize # make space for CLS token, shape (630, 127)
        if self.model_name == 'conch_ViT-B-16':
            text_latent, token_emb = self._get_text_feat(text_after_token_emb=text_after_token_emb, text_after_tokenize=text_after_tokenize,
                                                         use_text_projection=True)
        else:
            # === OPENCLIP ===
            text_latent, token_emb = self._get_text_feat_openclip(text_after_token_emb=text_after_token_emb, text_after_tokenize=text_after_tokenize,
                                                         use_text_projection=True)
            

        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent

        return text_latent, token_emb

    def _encode_text(self, text, normalize=True, embed_cls=True):
        '''
        Copy from `conch/open_clip_custom/coca_model.py`

        **For hard prompts**

        when only use visual trunk

        By default, do normalize (True) and embed_cls (True)

        Input:
        - text: is text tokens (630, 128)
        '''
        text = text[:, :-1] if embed_cls else text # make space for CLS token, shape (630, 127)

        if self.model_name == 'conch_ViT-B-16':
            text_latent, token_emb = self._get_text_feat(text_after_tokenize=text)
        else:
            text_latent, token_emb = self._get_text_feat_openclip(text_after_tokenize=text)

        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent

        return text_latent, token_emb

    def _get_text_feat(self, text_after_token_emb=None, text_after_tokenize=None, use_text_projection=False):
        '''
        Code from the `forward` function of `conch/open_clip_custom/transformer.py`

        All of conch COCA text branch

        Inputs:
        - text: shape (630, 127)

        With soft prompts case:
        - text_after_token_emb: shape(5, 128, 768)
        - text_after_tokenize: shape(5, 127)
        '''
        seq_len = text_after_tokenize.shape[1] # 127
        cast_dtype = self.model.text.transformer.get_cast_dtype() # torch.float32
        if text_after_token_emb is None:
            x = self.model.text.token_embedding(text_after_tokenize).to(cast_dtype)  # (630, 127, 768) [batch_size, n_ctx, d_model]
        else:
            x = text_after_token_emb # soft prompt

        attn_mask = self.model.text.attn_mask # shape (128, 128)
        if self.model.text.cls_emb is not None: # torch.nn.parameter.Parameter, shape (768)
            seq_len += 1 # 128
            x = torch.cat([x, self.model.text._repeat(self.model.text.cls_emb, x.shape[0])], dim=1) # self.model.text._repeat(self.model.text.cls_emb, x.shape[0]).shape = (630, 1, 768) -> x.shape: (630, 128, 768)
            cls_mask = self.model.text.build_cls_mask(text_after_tokenize, cast_dtype) # shape (7560, 128, 128)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len] # attn_mask[None, :seq_len, :seq_len].shape (1, 128, 128) and cls_mask[:, :seq_len, :seq_len].shape (7560, 128, 128) -> attn_mask.shape (7560, 128, 128)

        x = x + self.model.text.positional_embedding[:seq_len].to(cast_dtype) # x.shape = (630, 128, 768) and self.model.text.positional_embedding[:seq_len].shape = (128, 768) -> x.shape (630, 128, 768)
        x = x.permute(1, 0, 2)  # NLD -> LND, shape (128, 630, 768)
        x = self.model.text.transformer(x, attn_mask=attn_mask) # shape (128, 630, 768)
        x = x.permute(1, 0, 2)  # LND -> NLD, shape (630, 128, 768)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.model.text.cls_emb is not None: # torch.nn.parameter.Parameter, shape (768)
            pooled, tokens = x[:, -1], x[:, :-1] # pooled.shape = (630, 768), tokens.shape = (630, 127, 768)
            pooled = self.model.text.ln_final(pooled) # `ln_final` is `LayerNorm((768,), eps=1e-05, elementwise_affine=True)`, shape (630, 768)
        else:
            x = self.model.text.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), text_after_tokenize.argmax(dim=-1)], x

        if use_text_projection:
            pooled = pooled @ self.model.text.text_projection

        return pooled, tokens # pooled.shape (630, 768) and tokens.shape (630, 127, 768)

    def _get_text_feat_openclip(self, text_after_token_emb=None, text_after_tokenize=None, use_text_projection=False):
        '''
        The function is from `open_clip/model.py`
        '''
        seq_len = text_after_tokenize.shape[1] #
        cast_dtype = self.model.transformer.get_cast_dtype()

        if text_after_token_emb is None:
            x = self.model.token_embedding(text_after_tokenize).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        else:
            x = text_after_token_emb # soft prompt

        attn_mask = self.model.attn_mask
        attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.model.positional_embedding[:seq_len].to(cast_dtype)
        x = self.model.transformer(x, attn_mask=attn_mask)
        x = self.model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        pooled, tokens = text_global_pool(x, text_after_tokenize, self.model.text_pool_type)

        if self.model.text_projection is not None and use_text_projection:
            if isinstance(self.model.text_projection, nn.Linear):
                pooled = self.model.text_projection(pooled)
            else:
                pooled = pooled @ self.model.text_projection

        return pooled, tokens

    def forward(self, x_cls, x, x_concepts=None, cls_id=None):
        '''
        - Inputs:
          cls token (category feature): x_cls.shape: (32, 1, 1024)
          remaining tokens (image feature): x.shape: (32, 196, 1024)
          x_concepts: can be original raw concepts `self.concept_raw` (630,) OR extracted embeddings `self.concepts` (630, 512)
        '''
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        # ------- Soft context prompts -------
        soft_prompt_embeddings = None
        if self.set_learnable_prompts:
            prompts, tokenized_prompts = self.prompt_learner(cls_id) # prompts.shape: (5, 127, 768), tokeninzed_prompts.shape: (5, 127)
            # cls_id: None
            soft_prompt_embeddings, _ = self._encode_learnable_prompts(prompts, tokenized_prompts) # (5, 512)

            # normalize features
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True) # shape: (84, 512)

        # ------- Hard concept prompts -------
        if self.model_name is not None:
            if x_concepts is not None:
                if self.concept_embeddings is None: # (1, 630, 512)
                    # Case (2): hard concepts in raw
                    if self.model_name == 'conch_ViT-B-16':
                        text_tokens = tokenize(texts=x_concepts, tokenizer=self.tokenizer).to(self.device) # (630, 128)
                    else:
                        # === OPENCLIP ===
                        text_tokens = self.tokenizer(x_concepts).to(self.device)

                    if self.only_visual_trunk:
                        with torch.inference_mode():
                            concept_embeddings, _ = self._encode_text(text_tokens)
                    else:
                        with torch.inference_mode(): # for both CONCH and OPENCLIP
                            concept_embeddings = self.model.encode_text(text_tokens) # shape: conch (630, 512); openclip (630, 768)

                    self.concept_embeddings = concept_embeddings[None, :, :]
                    torch.save(self.concept_embeddings, self.concept_embeddings_save_dir)
                
        # ----- combine soft context vectors and hard concept vectors -----
        if soft_prompt_embeddings is not None and self.concept_embeddings is not None:
            # self.class_names.index()
            # self.groupname2clsname
            # self.concept_embeddings
            # soft_prompt_embeddings
            # concept2cls

            combined_embeddings = []
            for concept_idx in range(self.concept_embeddings.shape[1]): # 630
                corr_parent_label_int = self.dict_of_mapping_from_conceptid_to_softpromptid[concept_idx]

                combined_embeddings.append(torch.cat((self.concept_embeddings[0, concept_idx], soft_prompt_embeddings[corr_parent_label_int]), 0))

            combined_embeddings = torch.stack(combined_embeddings) # (630, 1024)
            combined_embeddings = combined_embeddings[None, :, :]
        # -----------------------------------------------------------------

        out = 0.0
        out_group = 0.0
        if self.n_unsup_concepts > 0:  # unsupervised stream,  self.n_unsup_concepts = 0
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_cls, self.unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream, self.n_concepts = 13 # TODO: global concepts
            # if self.model_name == 'conch_ViT-B-16':
            self.concept_embeddings /= self.concept_embeddings.norm(dim=-1, keepdim=True)

            if soft_prompt_embeddings is not None and self.concept_embeddings is not None:
                out_n, out_group_n, concept_attn, qk_parent_class = self.concept_tranformer(x_cls, combined_embeddings) # out_n.shape: (32, 1, 200), concept_attn.shape: (32, 2, 1, 13); qk_parent_class.shape: (990, 5) 5 parent classes of glomcrawl
            else:
                out_n, out_group_n, concept_attn, qk_parent_class = self.concept_tranformer(x_cls, self.concept_embeddings) # out_n.shape: (32, 1, 200), concept_attn.shape: (32, 2, 1, 13)
            # else:
            #     out_n, concept_attn = self.concept_tranformer(x_cls, self.concepts) # out_n.shape: (32, 1, 200), concept_attn.shape: (32, 2, 1, 13)

            concept_attn = concept_attn.mean(1)  # average over heads, shape: (32, 1, 13), glomcrawl (990, 1, 630)

            if out_n is not None:
                out = out + out_n.squeeze(1)  # squeeze token dimension, shape: (32, 200)
            else:
                out = None

            if out_group_n is not None:
                out_group = out_group + out_group_n.squeeze(1)  # squeeze token dimension, shape: (32, 200)
            else:
                out_group = None
            # out = out + out_n.mean(1)  # pool tokens sequence, shape: (32, 200)

        if self.n_spatial_concepts > 0:  # Spatial stream, self.n_spatial_concepts = 95
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, self.spatial_concepts) # out_s.shape: (32, 196, 200), spatial_concept_attn.shape: (32, 2, 196, 95)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads, shape: (32, 196, 95)
            out = out + out_s.mean(1)  # pool tokens sequence, shape: (32, 200)

        return out, out_group, unsup_concept_attn, concept_attn, spatial_concept_attn, qk_parent_class



def text_global_pool(x, text=None, pool_type='argmax'):
    '''
    Only for openclip-related functions
    The function is from openclip repo: `open_clip/transformer.py`

    Inputs:
    - text (torch.Tensor):
    - pool_type (str): default to 'argmax'

    '''
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x
 
    return pooled, tokens