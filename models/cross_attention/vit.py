import math

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from .ctc import CrossAttention
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer


def cub_cvit(backbone_name="vit_large_patch16_224", baseline=False, *args, **kwargs):
    """
    Args:
        baseline (bool): If true it returns the baseline model, which in this case it just the vit backbone without concept transformer
    """
    if not baseline:
        return CVIT(
            model_name=backbone_name,
            num_classes=200,
            n_unsup_concepts=0,
            n_concepts=13,
            n_spatial_concepts=95,
            num_heads=12,
            attention_dropout=0.1,
            projection_dropout=0.1,
            *args,
            **kwargs,
        )
    else:
        return ExplVIT(model_name=backbone_name, num_classes=200)


class ExplVIT(VisionTransformer):
    """VIT modified to return dummy concept attentions"""
    def __init__(self, num_classes=200, model_name="vit_base_patch16_224", *args, **kwargs):
        super().__init__(num_classes=num_classes)

        loaded_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.load_state_dict(loaded_model.state_dict())

    def forward(self, x):
        out = super().forward(x)
        return out, None, None, None


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
        *args,
        **kwargs,
    ):
        super().__init__()

        self_model_name = model_name

        if self.model_name == 'vit_base_patch16_224':
            self.feature_extractor = timm.create_model(self.model_name, pretrained=True, num_classes=num_classes) # CoCa model.visual
            del self.feature_extractor.head
        elif self.model_name == 'conch_ViT-B-16':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.feature_extractor, self.preprocess = create_model_from_pretrained(self.model_name, ckpt_path, device=device)
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
            *args,
            **kwargs, # {'batch_size': 32, 'debug': False, 'data_dir': '/data/hqvo3/data/cub2011', 'finetune_unfreeze_epoch': 0, 'num_workers': 8, 'seed': 42, 'data_name': 'CUB2011Parts'}
        )

    def forward(self, x):
        '''
        Inputs:
        - Original images - x.shape: (batch_size, 3, 224, 224)
        OR
        - Extracted embeds - x.shape: (batch_size, 512)
        '''
        if x.ndim == 4:
            x = self.feature_extractor.patch_embed(x) # (32, 196, 1024), TODO: CoCa model.visual.trunk.patch_embed: Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
        elif x.ndim == 2: # Use extracted embeddings
            x = x[:, None, :]

        cls_token = self.feature_extractor.cls_token.expand( # self.feature_extractor.cls_token.shape: (1, 1, 1024)
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks, cls_token.shape: (32, 1, 1024), TODO: CoCa model.visual.trunk.cls_token (torch.nn.parameter.Parameter) with shape (1, 1, 768)

        # self.feature_extractor.dist_token.shape: (1, 1, 1024)
        if self.feature_extractor.dist_token is None: # yes, dist token is None, TODO: CoCa has no dist_token too: model.visual.trunk.dist_token
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
            x[:, 0].unsqueeze(1), x[:, 1:] # shape: cls token: (32, 1, 1024) and remaining: (32, 196, 1024)
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
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model_name = model_name
        if self.model_name == 'conch_ViT-B-16':
            # ----- only for CONCH -----
            self.tokenizer = get_tokenizer()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = create_model_from_pretrained(self.model_name, ckpt_path, device=device)
            # _ = model.eval()
            # ----- only for CONCH -----

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
                if x_concepts.ndim == 1:
                    # Case (2): hard concepts in raw
                    text_tokens = tokenize(texts=x_concepts, tokenizer=self.tokenizer).to(device)
                    with torch.inference_mode(): # !! do we need to use this when inside forward !!
                        txt_feat = self.model.encode_text(text_tokens) # shape: (630, 512)
                    txt_feat = txt_feat[None, :, :]
                elif x_concepts.ndim == 2:
                    # Case (3) hard concepts in the form of extracted embeddings
                    txt_feat = x_concepts[None, :, :]
            else:
                # --- Soft prompts ---
                # Case (2): hard concepts in raw
                text_tokens = tokenize(texts=self.concepts, tokenizer=self.tokenizer).to(device)
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
                out_n, concept_attn = self.concept_tranformer(x_cls, txt_feat) # out_n.shape: (32, 1, 200), concept_attn.shape: (32, 2, 1, 13)
            else:
                out_n, concept_attn = self.concept_tranformer(x_cls, self.concepts) # out_n.shape: (32, 1, 200), concept_attn.shape: (32, 2, 1, 13)
            concept_attn = concept_attn.mean(1)  # average over heads, shape: (32, 1, 13)
            out = out + out_n.squeeze(1)  # squeeze token dimension, shape: (32, 200)

        if self.n_spatial_concepts > 0:  # Spatial stream, self.n_spatial_concepts = 95
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, self.spatial_concepts) # out_s.shape: (32, 196, 200), spatial_concept_attn.shape: (32, 2, 196, 95)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads, shape: (32, 196, 95)
            out = out + out_s.mean(1)  # pool tokens sequence, shape: (32, 200)

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn
