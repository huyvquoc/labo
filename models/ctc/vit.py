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


def cub_cvit(cfg, baseline=False, *args, **kwargs):
    """
    Args:
        baseline (bool): If true it returns the baseline model, which in this case it just the vit backbone without concept transformer
    """
    if not baseline:
        return CVIT(
            model_name=cfg.clip_model,
            num_classes=cfg.num_cls,
            n_unsup_concepts=cfg.num_unsup_concept,
            n_concepts=cfg.num_concept,
            n_spatial_concepts=cfg.num_spatial_concept,
            num_heads=cfg.num_heads,
            attention_dropout=cfg.attention_dropout,
            projection_dropout=cfg.projection_dropout,
            ckpt_path=cfg.ckpt_path,
            concept_embeddings_save_dir=cfg.concept_embeddings_save_dir,
            use_extracted_img_embs=cfg.use_extracted_img_embs,
            only_visual_trunk=cfg.only_visual_trunk,
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

def extract_img_embs_using_conch_visual_trunk(list_of_img_paths, list_of_labels, dataset_class, transform_compose, only_visual_trunk=False, ckpt_path='/data/hqvo3/weights/CONCH/pytorch_model.bin', do_aug=False, n_iters=10):
    '''
    This function is mostly used in the data module file

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
    feature_extractor, preprocess = create_model_from_pretrained('conch_ViT-B-16', ckpt_path, device=device)
    if only_visual_trunk:
        feature_extractor = feature_extractor.visual.trunk

    list_of_img_embs = []
    list_of_new_labels = [] # only change when there are augmentations
    if do_aug:
      for idx in range(n_iters):
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
        num_classes=200,
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
        only_visual_trunk=False,
        *args,
        **kwargs,
    ):
        '''
        Inputs:
        - only_visual_trunk: whether to only use embedding from visual trunk of CONCH, or use the one from attn_contrast
        '''
        super().__init__()

        self.model_name = model_name

        if self.model_name == 'vit_base_patch16_224':
            self.feature_extractor = timm.create_model(self.model_name, pretrained=True, num_classes=num_classes) # CoCa model.visual
            del self.feature_extractor.head
        elif self.model_name == 'conch_ViT-B-16':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.feature_extractor, self.preprocess = create_model_from_pretrained(self.model_name, ckpt_path, device=device)
            if only_visual_trunk:
                self.feature_extractor = self.feature_extractor.visual.trunk
            # _ = self.feature_extractor.eval()

        self.classifier = ConceptTransformerVIT(
            embedding_dim=self.feature_extractor.embed_dim, # 1024
            num_classes=num_classes, # 200
            attention_dropout=attention_dropout, # 0.1
            projection_dropout=projection_dropout, # 0.1
            n_unsup_concepts=n_unsup_concepts, # 0
            n_concepts=n_concepts, # 13
            n_spatial_concepts=n_spatial_concepts, # 95
            model_name=self.model_name,
            ckpt_path=ckpt_path,
            concept_embeddings_save_dir=concept_embeddings_save_dir,
            *args,
            **kwargs, # {'batch_size': 32, 'debug': False, 'data_dir': '/data/hqvo3/data/cub2011', 'finetune_unfreeze_epoch': 0, 'num_workers': 8, 'seed': 42, 'data_name': 'CUB2011Parts'}
        )

        self.use_extracted_img_embs = use_extracted_img_embs

    def forward(self, x, x_concepts):
        '''
        Inputs:
        - Original images - x.shape: (batch_size, 3, 224, 224)
        - x_concepts: can be original raw concepts `self.concept_raw` (630,) OR extracted embeddings `self.concepts` (630, 512)
        OR
        - Extracted embeds - x.shape: (batch_size, 512) or (batch_size, 768) for conch.vision.trunk
        '''
        if not self.use_extracted_img_embs:
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

        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.classifier(
            x[:, 0].unsqueeze(1), x[:, 1:], x_concepts # shape: cls token: (32, 1, 1024) and remaining: (32, 196, 1024)
        )
        # out.shape: (32, 200)
        # unsup_concept_attn: None
        # concept_attn.shape: (32, 1, 13)
        # spatial_concept_attn.shape: (32, 196, 95)
        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


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
        embedding_dim=768,
        num_classes=10,
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
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model_name = model_name
        self.only_visual_trunk = only_visual_trunk
        if self.model_name == 'conch_ViT-B-16':
            # ----- only for CONCH -----
            self.tokenizer = get_tokenizer()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model, self.preprocess = create_model_from_pretrained(self.model_name, ckpt_path, device=self.device)
            # _ = self.model.eval()
            # ----- only for CONCH -----

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
            self.concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
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

    def _encode_text(self, text, normalize=True, embed_cls=True):
        '''
        Copy from `conch/open_clip_custom/coca_model.py`

        By default, do normalize (True) and embed_cls (True)

        Input:
        - text: is text tokens
        '''
        text = text[:, :-1] if embed_cls else text # make space for CLS token
        text_latent, token_emb = self._get_text_feat(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent

        return text_latent, token_emb

    def _get_text_feat(self, text):
        '''
        All of conch COCA text branch
        '''
        cast_dtype = self.model.text.transformer.get_cast_dtype()
        seq_len = text.shape[1]
        x = self.model.text.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.model.text.attn_mask
        if self.model.text.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self.model.text._repeat(self.model.text.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.model.text.build_cls_mask(text, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.model.text.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.text.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.model.text.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.model.text.ln_final(pooled)
        else:
            x = self.model.text.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x

        return pooled, tokens

    def forward(self, x_cls, x, x_concepts=None):
        '''
        - Inputs:
          cls token (category feature): x_cls.shape: (32, 1, 1024)
          remaining tokens (image feature): x.shape: (32, 196, 1024)
          x_concepts: can be original raw concepts `self.concept_raw` (630,) OR extracted embeddings `self.concepts` (630, 512)
        '''
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        if self.model_name == 'conch_ViT-B-16':
            if x_concepts is not None:
                # --- Hard concept prompts ---
                if self.concept_embeddings is None:
                    # Case (2): hard concepts in raw
                    text_tokens = tokenize(texts=x_concepts, tokenizer=self.tokenizer).to(self.device)
                    if self.only_visual_trunk:
                        with torch.inference_mode():
                            concept_embeddings, _ = self._encode_text(text_tokens)
                    else:
                        with torch.inference_mode(): # !! do we need to use this when inside forward !!
                            concept_embeddings = self.model.encode_text(text_tokens) # shape: (630, 512)

                    self.concept_embeddings = concept_embeddings[None, :, :]
                    torch.save(self.concept_embeddings, self.concept_embeddings_save_dir)
                
                # elif x_concepts.ndim == 2:
                #     # Case (3) hard concepts in the form of extracted embeddings
                #     txt_feat = x_concepts[None, :, :]
            else:
                # --- Soft prompts ---
                # Case (2): hard concepts in raw
                text_tokens = tokenize(texts=self.concepts, tokenizer=self.tokenizer).to(self.device)
                with torch.inference_mode(): # !! do we need to use this when inside forward !!
                    txt_feat = self.model.encode_text(text_tokens) # shape: (630, 512)
                txt_feat = txt_feat[None, :, :]
                # TODO: either set txt_feat to learnable here using nn.Parameter()


        out = 0.0
        if self.n_unsup_concepts > 0:  # unsupervised stream,  self.n_unsup_concepts = 0
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_cls, self.unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream, self.n_concepts = 13 # TODO: global concepts
            if self.model_name == 'conch_ViT-B-16':
                out_n, concept_attn = self.concept_tranformer(x_cls, self.concept_embeddings) # out_n.shape: (32, 1, 200), concept_attn.shape: (32, 2, 1, 13)
            else:
                out_n, concept_attn = self.concept_tranformer(x_cls, self.concepts) # out_n.shape: (32, 1, 200), concept_attn.shape: (32, 2, 1, 13)
            concept_attn = concept_attn.mean(1)  # average over heads, shape: (32, 1, 13)
            out = out + out_n.squeeze(1)  # squeeze token dimension, shape: (32, 200)

        if self.n_spatial_concepts > 0:  # Spatial stream, self.n_spatial_concepts = 95
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, self.spatial_concepts) # out_s.shape: (32, 196, 200), spatial_concept_attn.shape: (32, 2, 196, 95)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads, shape: (32, 196, 95)
            out = out + out_s.mean(1)  # pool tokens sequence, shape: (32, 200)

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn
