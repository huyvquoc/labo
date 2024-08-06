import collections
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer


# _tokenizer = _Tokenizer()

class HierarchyAwarePromptLearner(nn.Module):
    '''
      Group names are sorted in alphabetical order
      - G1: Glomerular Basement Membrane (GBM), Bowman's Capsule, Epithelial Cells
      - G2: Mesangial and Endocapillary Lesions
      - G3: Miscellaneous Lesions and Key Normal Structures
      - G4: Necrotizing and Crescentic Lesions
      - G5: Tuft Sclerosis and Collapse
    '''
    def __init__(self, cfg, classnames, groupname2clsname, clip_model, tokenizer, use_group=True, multilabel_on=False):
        '''
        Initialize learnable prompts in ConceptTransformerVIT

        Inputs:
        - Init `clip_model` with `self.model` in ConceptTransformerVIT
        - Init `tokenizer` with `self.tokenizer` in ConceptTransformerVIT

        Outputs of `forward`:
        - prompts: (84, 77, 512)
        - tokenized_prompts: (84, 77)
        '''
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.multilabel_on = multilabel_on

        self.groupname2clsname = collections.OrderedDict(sorted(groupname2clsname.items()))
        self.groupnames = list(groupname2clsname.keys())

        if use_group:
            self.n_cls = len(self.groupnames) # 5
        else:
            self.n_cls = len(classnames) # 42

        self.n_ctx_pos = cfg.n_ctx_pos # 16 TODO: what is it?
        self.n_ctx_neg = cfg.n_ctx_neg # 16
        ctx_init_pos = cfg.positive_prompt_init.strip() # '' (empty string)
        ctx_init_neg = cfg.negative_prompt_init.strip() # '' (empty string)

        if cfg.clip_model == 'conch_ViT-B-16':
            # dtype = clip_model.dtype # torch.float32
            dtype = clip_model.text.transformer.get_cast_dtype() # TODO: if the above does not work
            ctx_dim = clip_model.text.ln_final.weight.shape[0] # 512
        else:
            dtype = clip_model.dtype # torch.float32
            ctx_dim = clip_model.ln_final.weight.shape[0] # 512

        if ctx_init_pos and ctx_init_neg:
            # If there are initial strings
            # --- Inject hard prompt => need to extract token embedding to have soft prompt to be optimized ---
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            self.n_ctx_pos = len(ctx_init_pos.split(" "))
            self.n_ctx_neg = len(ctx_init_neg.split(" "))
            if cfg.clip_model == 'conch_ViT-B-16':
                prompt_pos = tokenize(texts=ctx_init_pos, tokenizer=tokenizer)[:, :-1].to(self.device) # --- CONCH ---
                prompt_neg = tokenize(texts=ctx_init_neg, tokenizer=tokenizer)[:, :-1].to(self.device) # --- CONCH ---
            else:
                prompt_pos = tokenizer(ctx_init_pos)[:, :-1].to(self.device)
                prompt_neg = tokenizer(ctx_init_neg)[:, :-1].to(self.device)

            with torch.no_grad():
                if cfg.clip_model == 'conch_ViT-B-16':
                    embedding_pos = clip_model.text.token_embedding(prompt_pos).type(dtype) # --- CONCH ---
                    embedding_neg = clip_model.text.token_embedding(prompt_neg).type(dtype) # --- CONCH ---
                else:
                    embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                    embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)

            ctx_vectors_pos = embedding_pos[0, 1: 1 + self.n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + self.n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if cfg.csc:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos)) # the same extracted text embedding (positive) for each class
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg)) # the same extracted text embedding (negative) for each class
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # --- Init empty tensor for soft prompt  ---
            # Random Initialization
            if cfg.csc:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(self.n_cls, self.n_ctx_pos, ctx_dim, dtype=dtype) # clip (42, 16, 512, torch.float32); conch group (5, 16, 512)
                ctx_vectors_neg = torch.empty(self.n_cls, self.n_ctx_neg, ctx_dim, dtype=dtype) # clip (42, 16, 512, torch.float32); conch group (5, 16, 512)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(self.n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(self.n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * self.n_ctx_pos) # 'X X X X X X X X X X X X X X X X'
            prompt_prefix_neg = " ".join(["X"] * self.n_ctx_neg)

        print(f'Initial positive context: "{prompt_prefix_pos}"') # 'X X X X X X X X X X X X X X X X'
        print(f'Initial negative  context: "{prompt_prefix_neg}"') # 'X X X X X X X X X X X X X X X X'
        print(f"Number of positive context words (tokens): {self.n_ctx_pos}") # 16
        print(f"Number of negative context words (tokens): {self.n_ctx_neg}") # 16

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized, soft prompt clip, shape(42, 16, 512); conch group shape(5, 16, 512)
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized, soft prompt clip, shape(42, 16, 512); conch group shape(5, 16, 512)

        if use_group:
            self.classnames = [name.replace("_", " ") for name in self.groupnames]
        else:
            self.classnames = [name.replace("_", " ") for name in classnames]

        if cfg.clip_model == 'conch_ViT-B-16':
            name_lens = []
            for name in self.classnames:
                if cfg.clip_model == 'conch_ViT-B-16':
                    name_lens.append(len(tokenize(texts=[name], tokenizer=tokenizer)[:, :-1])) # ref: class name embeddings, `/conch/downstream/zeroshot_path.py`, --- CONCH ---
                    # clip_model.encode_text(tokenize(texts=name, tokenizer=tokenizer).to(self.device)) # TODO: do we need to use `encode_text`?,
                else:
                    name_lens.append(len(tokenizer(name)[:, :-1])) # ref: class name embeddings
        # else:
        #     name_lens = [len(_tokenizer.encode(name)) for name in self.classnames] # one class name is converted into multiple token ids

        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in self.classnames] # each element is "X X X ... class name"
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in self.classnames] # the same as prompt pos

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        for p_pos, p_neg in zip(prompts_pos, prompts_neg):
            if cfg.clip_model == 'conch_ViT-B-16':
                tokenized_prompts_pos.append(tokenize(texts=[p_pos], tokenizer=tokenizer)[:, :-1].to(self.device)) # --- CONCH --- 
                tokenized_prompts_neg.append(tokenize(texts=[p_neg], tokenizer=tokenizer)[:, :-1].to(self.device)) # --- CONCH --- 
            else:
                tokenized_prompts_pos.append(tokenizer(p_pos)[:, :-1].to(self.device))
                tokenized_prompts_neg.append(tokenizer(p_neg)[:, :-1].to(self.device))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos) # clip (42, 77) 77 is the number of tokens after using clip.tokenize, conch group: shape(5, 127)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg) # clip (42, 77), conch group: shape(5, 127)
        with torch.no_grad():
            if cfg.clip_model == 'conch_ViT-B-16':
                embedding_pos = clip_model.text.token_embedding(tokenized_prompts_pos).type(dtype) # clip (42, 77, 512); conch group (5, 127, 512), --- CONCH ---
                embedding_neg = clip_model.text.token_embedding(tokenized_prompts_neg).type(dtype) # clip (42, 77, 512); conch group (5, 127, 512), --- CONCH ---
            else:
                embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype) # clip
                embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype) # clip

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :]) # clip (42, 1, 512); conch group (5, 1, 768)
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + self.n_ctx_pos:, :]) # clip (42, 60, 512); conch group (5, x, 512)
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :]) # clip (42, 1, 512); conch group (5, 1, 768)
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + self.n_ctx_neg:, :]) # clip (42, 60, 512); conch group (5, x, 512)

        if self.multilabel_on:
            tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)  # torch.Tensor, clip (84, 77) = (2*42, 77), conch group (10, 127) = (2*5, 127)
        else:
            tokenized_prompts = tokenized_prompts_pos
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens # [7, 6, 10, 3, 4, 7, 4, 7, 6, 3, 3, 4, 9, 5, ...]
    
    def get_groupname2clsname(self):
        return self.groupnames, self.groupname2clsname

    def forward(self, cls_id=None):
        '''
        Inputs:
        - soft prompts
        - cls_id: for zero shot experiment, check DualCoOp

        Outputs:
        - prompts
        - tokenized prompts
        '''
        ctx_pos = self.ctx_pos # soft prompt, nn.Parameter, clip shape(42, 16, 512); conch group shape(5, 16, 512)
        ctx_neg = self.ctx_neg # soft prompt, nn.Parameter, clip shape(42, 16, 512); conch group shape(5, 16, 512)

        if ctx_pos.dim() == 2:
            if cls_id is None:
                ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_pos = ctx_pos.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_pos = ctx_pos[cls_id]

        if ctx_neg.dim() == 2:
            if cls_id is None:
                ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_neg = ctx_neg.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_neg = ctx_neg[cls_id]

        if cls_id is None:
            prefix_pos = self.token_prefix_pos # "X X X ... {cls name}" after going through tokenize and token embedding (the first part of text branch), (42, 1, 512)
            prefix_neg = self.token_prefix_neg # "X X X ... {cls name}" after going through tokenize and token embedding (the first part of text branch), (42, 1, 512)
            suffix_pos = self.token_suffix_pos # "X X X ... {cls name}" after going through tokenize and token embedding (the first part of text branch), (42, 60, 512)
            suffix_neg = self.token_suffix_neg # "X X X ... {cls name}" after going through tokenize and token embedding (the first part of text branch), (42, 60, 512)
        else:
            prefix_pos = self.token_prefix_pos[cls_id]
            prefix_neg = self.token_prefix_neg[cls_id]
            suffix_pos = self.token_suffix_pos[cls_id]
            suffix_neg = self.token_suffix_neg[cls_id]


        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_cls, 1, dim) = (42, 1, 512), text token embeddings of 'X X X ... cls name', why split into prefix and suffix
                ctx_pos,  # (n_cls, n_ctx, dim) = (42, 16, 512), soft prompt
                suffix_pos,  # (n_cls, *, dim) = (42, 60, 512), text token embeddings of 'X X X ... cls name'
            ],
            dim=1,
        ) # (42, 77, 512): 77 is sum of second dim of the 3 tensor above; conch coca text branch (5, 128, 768)

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=1,
        ) # (42, 77, 512); conch coca text branch (5, 128, 768)

        if self.multilabel_on:
            prompts = torch.cat([prompts_neg, prompts_pos], dim=0) # (84, 77, 512) both positive and negative
        else:
            prompts = prompts_pos # conch: (5, 128, 768)

        if self.multilabel_on:
            if cls_id is not None:
                tokenized_prompts_pos = self.tokenized_prompts[self.n_cls:][cls_id]
                tokenized_prompts_neg = self.tokenized_prompts[:self.n_cls][cls_id]
                tokenized_prompts = torch.cat([tokenized_prompts_neg, tokenized_prompts_pos], dim=0)
            else:
                tokenized_prompts = self.tokenized_prompts # still "X X X ... {cls name}", token after clip.tokenize step (not do token embedding), (84, 77)
        else:
            tokenized_prompts = self.tokenized_prompts # (5, 128)


        return prompts, tokenized_prompts # after text token embeddings hard-soft-hard (84, 77, 512) (5, 128, 768) and after clip.tokenize (84, 77) (5, 128)
        # TODO: don't return token embedding `prompts`


class TextEncoder(nn.Module):
    '''
    From DualCoOp. Not used yet
    '''
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding # shape: (77, 512)
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection # torch.nn.Parameter, shape (512, 512)
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        '''
        prompts.shape: (84, 77, 512)
        tokenized_prompts: (84, 77)

        Inputs:
        - prompts: shape (batch size, #tokens, text embedding dim) = (84, 77, 512)

        Outputs:
        - x: shape (batch size, text embedding dim) = (84, 512)
        '''
        x = prompts + self.positional_embedding.type(self.dtype) # (84, 77, 512) = (2*42, 77, 512)
        x = x.permute(1, 0, 2)  # NLD -> LND, shape: (77, 84, 512)
        x = self.transformer(x) # (77, 84, 512)
        x = x.permute(1, 0, 2)  # LND -> NLD, (84, 77, 512)
        x = self.ln_final(x).type(self.dtype) # (84, 77, 512)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # torch.arange(x.shape[0]): tensor list from 0 -> 83
        # tokenized_prompts.argmax(dim=-1): list of different 2x numbers, have the same length as torch.arange, shape (84)
        # x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)].shape: (84, 512)
        # self.text_projection: tensor, shape (512, 512)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x # (84, 512) = (2*42, 512)