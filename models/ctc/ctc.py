# Cross-attention code based on https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cct.py by Phil Wang
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# Pre-defined CTC Models
__all__ = ["mnist_ctc"]


def mnist_ctc(*args, **kwargs):
    return _cct(
        num_layers=2,
        num_heads=2,
        mlp_ratio=1,
        embedding_dim=128,
        img_size=28,
        n_input_channels=1,
        num_classes=2,
        n_unsup_concepts=0,
        n_concepts=10,
        n_spatial_concepts=0,
        *args,
        **kwargs,
    )


def tmnist_ctc(*args, **kwargs):
    return _cct(
        num_layers=2,
        n_conv_layers=2,
        num_heads=2,
        mlp_ratio=1,
        embedding_dim=256,
        n_input_channels=1,
        num_classes=10,
        n_concepts=0,
        n_spatial_concepts=11,
        pooling_kernel_size=4,
        pooling_stride=3,
        pooling_padding=2,
        max_seqlen=4,
        *args,
        **kwargs,
    )




def _cct(
    num_layers,
    num_heads,
    mlp_ratio,
    embedding_dim,
    kernel_size=3,
    stride=None,
    padding=None,
    *args,
    **kwargs,
):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1) # stride = 1
    padding = padding if padding is not None else max(1, (kernel_size // 2)) # padding = 1
    return CTC(
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        embedding_dim=embedding_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        *args,
        **kwargs,
    )


# Modules
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads # 2
        head_dim = dim // self.num_heads # head_dim = 64
        self.scale = head_dim ** -0.5 # 0.125

        self.qkv = nn.Linear(dim, dim * 3, bias=False) # dim = 128, 128*3 = 384
        self.attn_drop = nn.Dropout(attention_dropout) # attention_dropout = 0.1
        self.proj = nn.Linear(dim, dim) # dim = 128
        self.proj_drop = nn.Dropout(projection_dropout) # projection_dropout = 0.1

    def forward(self, x):
        B, N, C = x.shape # 32, 196, 128
        qkv = (
            self.qkv(x) # (32, 196, 384)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads) # (32, 196, 3, 2, 64)
            .permute(2, 0, 3, 1, 4) # (3, 32, 2, 196, 64)
        )
        q, k, v = qkv[0], qkv[1], qkv[2] # q.shape = k.shape = v.shape: (32, 2, 196, 64)

        # k.transpose(-2, -1).shape: (32, 2, 64, 196)
        attn = (q @ k.transpose(-2, -1)) * self.scale # shape: (32, 2, 196, 196)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (attn @ v).shape: (32, 2, 196, 64)
        # (attn @ v).transpose(1, 2).shape: (32, 196, 2, 64)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self, dim, n_outputs=None, n_group_outputs=None, num_heads=8, attention_dropout=0.1, projection_dropout=0.0, dict_of_mapping_from_conceptid_to_softpromptid=None, use_child_classes=True, use_parent_classes=False,
        use_attn_parent_branch=False,
    ): # dim = 128, n_outputs = 2, num_heads = 2, attention_dropout = 0.1, projection_dropout = 0.1
        super().__init__()
        n_outputs = n_outputs if n_outputs else dim
        self.num_heads = num_heads
        head_dim = dim // self.num_heads # 64
        self.scale = head_dim ** -0.5 # 0.125

        self.q = nn.Linear(dim, dim, bias=False) # Linear(in_features=512, out_features=512, bias=False)
        self.kv = nn.Linear(dim * 2, dim * 2, bias=False) # Linear(in_features=512, out_features=1024, bias=False)
        # nn.init.xavier_uniform_(self.q.weight)
        # nn.init.xavier_uniform_(self.kv.weight)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.dict_of_mapping_from_conceptid_to_softpromptid = dict_of_mapping_from_conceptid_to_softpromptid # one-to-one
        self.dict_of_mapping_from_softpromptid_to_conceptids = {} # one-to-many
        for key_conceptid, value_softpromptid in self.dict_of_mapping_from_conceptid_to_softpromptid.items():
            if value_softpromptid not in self.dict_of_mapping_from_softpromptid_to_conceptids:
                self.dict_of_mapping_from_softpromptid_to_conceptids[value_softpromptid] = []
            self.dict_of_mapping_from_softpromptid_to_conceptids[value_softpromptid].append(key_conceptid) # this list has been sorted, `collections.OrderedDict` is used in `models/ctc/vit.py`

        self.use_child_classes = use_child_classes
        if self.use_child_classes:
            # self.fc1 = nn.Linear(dim, 256)
            # self.relu1 = nn.LeakyReLU(0.1)
            # nn.init.kaiming_normal_(self.fc1.weight)
            self.proj = nn.Linear(dim, n_outputs) # Linear(in_features=512, out_features=42, bias=True)
            # nn.init.xavier_uniform_(self.proj.weight)
            self.proj_drop = nn.Dropout(projection_dropout)

        self.use_parent_classes = use_parent_classes
        if self.use_parent_classes:
            self.proj2 = nn.Linear(dim, n_group_outputs)
            self.proj_drop2 = nn.Dropout(projection_dropout)

        self.use_attn_parent_branch = use_attn_parent_branch

    def forward(self, x, y):
        '''
        Inputs:
        - x (or x_pooled): image features, MNIST shape (32, 1, 128), CUB no spatial shape (32, 1, 1024), CUB spatial shape (32, 196, 1024), GlomCrawl (990, 1, 512)
        - y (or self.concepts): concepts, MNIST shape (1, 10, 128), CUB no spatial shape (1, 13, 1024), CUB spatial shape (1, 95, 1024), GlomCrawl with only concepts (1, 630, 512), GlomCrawl with concepts contexts (1, 630, 1024)
        '''
        B, Nx, C = x.shape # MNIST: 32, 1, 128; CUB no spatial: 32, 1, 1024; CUB spatial: 32, 196, 1024, GlomCrawl - one emb for an entire image: (990, 1, 512)
        By, Ny, Cy = y.shape # MNIST: 1, 10, 128; CUB no spatial: 1, 13, 1024; CUB spatial: 1, 95, 1024, GlomCrawl - one emb for an entire image: (1, 630, 512)

        # if not self.both_concepts_and_contexts:
        #     assert C == Cy, "Feature size of x and y must be the same"
        # else:
        #     assert 2*C == Cy, "Feature size of y must be twice as much as x"

        # --- MNIST ---
        # self.q(x).shape: (32, 1, 128)
        # self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).shape: (32, 1, 1, 2, 64)
        # self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).shape: (1, 32, 2, 1, 64)
        # --- CUB no spatial ---
        # self.q(x).shape: (32, 1, 1024)
        # self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).shape: (32, 1, 1, 2, 512)
        # self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).shape: (1, 32, 2, 1, 512) (placeholder 1, batch size, 2 heads, number of patches, embedding dim)
        # --- CUB spatial ---
        # self.q(x).shape: (32, 196, 1024)
        # self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).shape: (32, 196, 1, 2, 512)
        # self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).shape: (1, 32, 2, 196, 512)
        # --- GlomCrawl - one emb for an entire image ---
        # self.q(x).shape: (990, 1, 512)
        # self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).shape: (990, 1, 1, 8, 64)
        # self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).shape: (1, 990, 8, 1, 64)
        # --- GlomCrawl - multiple patch embs for an entire image ---
        q = self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = (
            self.kv(y) # shape: mnist: (1, 10, 256); cub no spatial: (1, 13, 2048)
            .reshape(By, Ny, 2, self.num_heads, C // self.num_heads) # shape: mnist: (1, 10, 2, 2, 64); cub no spatial: (1, 13, 2, 2, 512)
            .permute(2, 0, 3, 1, 4) # shape: mnist: (2, 1, 2, 10, 64); cub no spatial: (2, 1, 2, 13, 512) (key and value, placeholder 2, 2 heads, #contexts, embedding dim)
        )

        q = q[0] # mnist: (32, 2, 1, 64); cub no spatial: (32, 2, 1, 512); cub spatial: (32, 2, 196, 512), glomcrawl: (990, 8, 1, 64)
        k, v = kv[0], kv[1] 
        # k: mnist: (1, 2, 10, 64); cub no spatial: (1, 2, 13, 512); cub spatial: (1, 2, 95, 512), glomcrawl: (1, 8, 630, 64)
        # v: mnist: (1, 2, 10, 64); cub no spatial: (1, 2, 13, 512); cub spatial: (1, 2, 95, 512), glomcrawl: (1, 8, 630, 64)

        attn = (q @ k.transpose(-2, -1)) * self.scale # mnist: (32, 2, 1, 10); cub no spatial: (32, 2, 1, 13); cub spatial: (32, 2, 196, 95); glomcrawl: (990, 8, 1, 630) (batch size, num heads, number of patches, #concepts)
        attn = attn.softmax(dim=-1) # mnist: (32, 2, 1, 10); cub no spatial: (32, 2, 1, 13); cub spatial: (32, 2, 196, 95); glomcrawl: (990, 8, 1, 630), pairwise score between each patch and each concept
        attn = self.attn_drop(attn) # mnist: (32, 2, 1, 10); cub no spatial: (32, 2, 1, 13); cub spatial: (32, 2, 196, 95); glomcrawl: (990, 8, 1, 630)

        qk_parent_class = None
        if self.use_attn_parent_branch:
            # ----- reduce from (990, 8, 1, 630) to (990, 8, 1, 5) -----
            # self.dict_of_mapping_from_conceptid_to_softpromptid
        
            qk_parent_class = []
            for key_softpromptid, value_conceptids in self.dict_of_mapping_from_softpromptid_to_conceptids.items():
                indices = torch.tensor(value_conceptids).to('cuda')
                # index_select can be differentiable wrt source tensor but not indices (https://discuss.pytorch.org/t/problem-about-the-function-torch-index-select-input-dim-index-out-none/51668/7)
                attn_sub = torch.index_select(attn, 3, indices) # shape (990, 8, 1, #parent_class_related_concepts). Check differentiability following this post: https://discuss.pytorch.org/t/how-to-know-if-specific-function-is-differential-or-not/125960
                attn_sub = torch.squeeze(attn_sub) # (990, 8, #parent_class_related_concepts)
                attn_sub = torch.mean(attn_sub, dim=1, keepdim=False) # wrt #heads (990, #parent_class_related_concepts)
                attn_sub = torch.mean(attn_sub, dim=-1, keepdim=True) # wrt #concepts_of_current_parent_class (990, 1)
                qk_parent_class.append(attn_sub)
            qk_parent_class = torch.cat(qk_parent_class, dim=1)

            # ----------------------------------------------------------

        # MNIST: (attn @ v).shape: (32, 2, 1, 64); CUB no spatial: (32, 2, 1, 512); CUB spatial: (32, 2, 196, 512); glomcrawl: (990, 8, 1, 64)
        # MNIST: (attn @ v).transpose(1, 2).shape: (32, 1, 2, 64); CUB no spatial: (32, 1, 2, 512); CUB spatial: (32, 196, 2, 512); glomcrawl: (990, 1, 8, 64)
        # MNIST: (attn @ v).transpose(1, 2).reshape(B, Nx, C).shape: (32, 1, 128); CUB no spatial: (32, 1, 1024); CUB spatial: (32, 196, 1024); glomcrawl: (990, 1, 512)
        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C) # MNIST: (attn @ v).transpose(1, 2).reshape(B, Nx, C).shape: (32, 1, 128); CUB no spatial: (32, 1, 1024); CUB spatial: (32, 196, 1024)

        out1 = None
        if self.use_child_classes:
            # x = self.relu1(self.fc1(x))
            out1 = self.proj(x) # shape: mnist (32, 1, 2); cub no spatial: (32, 1, 200); cub spatial: (32, 196, 200); glomcrawl: (990, 1, 42)
            out1 = self.proj_drop(out1) # shape: (32, 1, 2); cub no spatial:  (32, 1, 200); cub spatial: (32, 196, 200); glomcrawl: (990, 1, 42)

        out2 = None
        if self.use_parent_classes:
            out2 = self.proj2(x)
            out2 = self.proj_drop2(out2)

        return out1, out2, attn, qk_parent_class #  x: (32, 1, 2), attn: (32, 2, 1, 10); glomcrawl with soft prompt: out1.shape(990, 1, 42), out2.shape(990, 1, 5)


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """

    def __init__(
        self,
        d_model, # 128
        nhead, # 2
        dim_feedforward=2048, # 128
        dropout=0.1, # 0.1
        attention_dropout=0.1, # 0.1
        drop_path_rate=0.1, # 0.0
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(
            dim=d_model, # 128
            num_heads=nhead, # 2
            attention_dropout=attention_dropout, # 0.1
            projection_dropout=dropout, # 0.1
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward) # d_model = 128, dim_feedforward = 128
        self.dropout1 = nn.Dropout(dropout) # dropout = 0.1
        self.norm1 = nn.LayerNorm(d_model) # d_model = 128
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # d_model = 128, dim_feedforward = 128 
        self.dropout2 = nn.Dropout(dropout) # dropout = 0.1

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity() # drop_path_rate = 0.0

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        '''
        Inputs:
        - src: (32, 196, 128)
        '''
        src = src + self.drop_path(self.self_attn(self.pre_norm(src))) # shape: (32, 196, 128)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PretrainedTokenizer(nn.Module):
    def __init__(
        self, name="resnet50", embedding_dim=768, flatten=False, freeze=True, *args, **kwargs
    ):
        """
        Note: if img_size=448, number of tokens is 14 x 14
        """
        super().__init__()

        self.flatten = flatten
        self.embedding_dim = embedding_dim

        ver = torchvision.__version__.split("a")[0]
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load("pytorch/vision:v" + ver, name, pretrained=True)
        model.eval()

        # Remove final pooler and classifier
        self.n_features = model.fc.in_features
        model.avgpool = nn.Sequential()
        model.fc = nn.Sequential()

        # Freeze model
        self.model = model
        if freeze:
            self.model.requires_grad_(False)

        self.fc = nn.Conv2d(self.n_features, embedding_dim, kernel_size=1)

    def forward(self, x):
        out = self.model(x)

        width = int(math.sqrt(out.shape[1] / self.n_features))
        out = out.unflatten(-1, (self.n_features, width, width))
        out = self.fc(out)

        if self.flatten:
            return out.flatten(-2, -1).transpose(-2, -1)
        else:
            return out


class Tokenizer(nn.Module):
    """Applies strided convolutions to the input and then tokenizes (creates patches)"""

    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_conv_layers=1,
        n_input_channels=3,
        n_output_channels=64,
        in_planes=64,
        activation=None, # torch.nn.modules.activation.ReLU
        max_pool=True,
        conv_bias=False,
        flatten=True,
    ):
        super().__init__()

        n_filter_list = (
            [n_input_channels] # 1
            + [in_planes for _ in range(n_conv_layers - 1)] # n_conv_layers = 1, in_planes = 64
            + [n_output_channels] # 128
        ) # n_filter_list = [1, 128] is a list

        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        n_filter_list[i],
                        n_filter_list[i + 1],
                        kernel_size=(kernel_size, kernel_size), # kernel_size = 3
                        stride=(stride, stride), # stride = 1
                        padding=(padding, padding), # padding = 1
                        bias=conv_bias, # Falsse
                    ),
                    nn.Identity() if activation is None else activation(),
                    nn.MaxPool2d(
                        kernel_size=pooling_kernel_size, # 3
                        stride=pooling_stride, # 2
                        padding=pooling_padding, # 1
                    )
                    if max_pool
                    else nn.Identity(),
                )
                for i in range(n_conv_layers)
            ]
        )

        self.flatten = flatten # True
        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        '''
        Args:
        - n_channels = 1
        - height = 28
        - width = 28
        '''
        out = self.forward(torch.zeros((1, n_channels, height, width)))
        if self.flatten: # True
            return out.shape[1] # 128
        else:
            return self.flattener(out).transpose(-2, -1).shape[1]

    def forward(self, x):
        '''
        Inputs:
        - x.shape: (32, 1, 28, 28)
        '''
        out = self.conv_layers(x) # out.shape: (32, 128, 14, 14)
        if self.flatten:
            # self.flattener(out).shape: (32, 128, 196)
            return self.flattener(out).transpose(-2, -1) # shape: (32, 196, 128)
        else:
            return out

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class FactorizedPositionEncoding(nn.Module):
    def __init__(self, max_seqlen, dim, embedding_dim):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seqlen

        # Create factorized position embeddings
        self.positional_emb = nn.ParameterList()
        for i in range(dim):
            self.positional_emb.append(
                nn.Parameter(
                    torch.zeros(1, self.max_seqlen, *(dim - 1) * [1], embedding_dim).transpose(
                        1, i + 1
                    ),
                    requires_grad=True,
                )
            )

    def forward(self, x):
        x = F.pad(
            x,
            [j for i in x.shape[-1:1:-1] for j in [0, self.max_seqlen - i]],
            mode="constant",
            value=0,
        )
        x = x.transpose(-1, 1)
        x = x + sum(self.positional_emb)
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim=768, # 128
        num_layers=12, # 2
        num_heads=12, # 2
        mlp_ratio=4.0, # 1
        dropout_rate=0.1, # 0.1
        attention_dropout=0.1, # 0.1
        stochastic_depth_rate=0.1, # 0.1
        positional_embedding="sine", # 'sine'
        sequence_length=None, # 196
        *args,
        **kwargs,
    ):
        super().__init__()

        positional_embedding = (
            positional_embedding
            if positional_embedding in ["sine", "learnable", "none"]
            else "sine"
        )
        dim_feedforward = int(embedding_dim * mlp_ratio) # dim_feedforward = 128
        self.embedding_dim = embedding_dim # embedding_dim = 128
        self.sequence_length = sequence_length # sequence_length = 196

        assert sequence_length is not None or positional_embedding == "none", (
            f"Positional embedding is set to {positional_embedding} and"
            f" the sequence length was not specified."
        )

        if positional_embedding != "none":
            if positional_embedding == "learnable":
                self.positional_emb = nn.Parameter(
                    torch.zeros(1, sequence_length, embedding_dim), requires_grad=True
                )
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(
                    self.sinusoidal_embedding(sequence_length, embedding_dim), requires_grad=False
                ) # sequence_length=196, embedding_dim=128
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout_rate) # dropout_rate=0.1
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        # stochastic_depth_rate = 0.1, num_layers = 2
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=embedding_dim, # 128
                    nhead=num_heads, # 2
                    dim_feedforward=dim_feedforward, # 128
                    dropout=dropout_rate, # 0.1
                    attention_dropout=attention_dropout, # 0.1
                    drop_path_rate=dpr[i], # [0.0, 0.10000000149011612]
                )
                for i in range(num_layers)
            ]
        )

        self.apply(self.init_weight)

    def forward(self, x):
        '''
        Inputs:
        - x.shape: (32, 196, 128)
        '''
        if self.positional_emb is not None:
            x += self.positional_emb # self.positional_emb.shape: (1, 196, 128)
        elif self.sequence_length is not None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode="constant", value=0)

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        '''
        n_channels = 196
        dim = 128
        '''
        pe = torch.FloatTensor(
            [[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)]
        )
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class ConceptTransformer(nn.Module):
    """
    Processes spatial and non-spatial concepts in parallel and aggregates the log-probabilities at the end
    """

    def __init__(
        self,
        embedding_dim=768, # 128
        num_classes=10, # 2
        num_heads=2, # 2
        attention_dropout=0.1, # 0.1
        projection_dropout=0.1, # 0.1
        n_unsup_concepts=0, # 0
        n_concepts=10, # 10
        n_spatial_concepts=0, # 0
        *args,
        **kwargs,
    ):
        super().__init__()

        # Unsupervised concepts
        self.n_unsup_concepts = n_unsup_concepts
        self.unsup_concepts = nn.Parameter(
            torch.zeros(1, n_unsup_concepts, embedding_dim), requires_grad=True
        ) # shape (1, 0 , 128)
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
        self.concepts = nn.Parameter(torch.zeros(1, n_concepts, embedding_dim), requires_grad=True) # (1, 10, 128)
        nn.init.trunc_normal_(self.concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_concepts > 0:
            self.concept_tranformer = CrossAttention(
                dim=embedding_dim, # 128
                n_outputs=num_classes, # 2
                num_heads=num_heads, # 2
                attention_dropout=attention_dropout, # 0.1
                projection_dropout=projection_dropout, # 0.1
            )

        # Sequence pooling for both non-spatial and unsupervised concepts
        if n_concepts > 0 or n_unsup_concepts > 0:
            self.token_attention_pool = nn.Linear(embedding_dim, 1)

        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        self.spatial_concepts = nn.Parameter(
            torch.zeros(1, n_spatial_concepts, embedding_dim), requires_grad=True
        ) # shape (1, 0, 128)
        nn.init.trunc_normal_(self.spatial_concepts, std=1.0 / math.sqrt(embedding_dim)) # std = 0.08838834764831843
        if n_spatial_concepts > 0:
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x):
        '''
        Inputs:
        - x.shape: (32, 196, 128)
        - self.n_unsup_concepts: 0 for MNIST odd/even
        - self.n_concepts: 10 for MNIST odd/even
        - self.n_spatial_concepts: 0
        - self.concepts: contains normalized values. Where are they from? Shape: (1, 10, 128)
        '''
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0
        if self.n_unsup_concepts > 0 or self.n_concepts > 0:
            token_attn = F.softmax(self.token_attention_pool(x), dim=1).transpose(-1, -2) # (32, 1, 196)
            x_pooled = torch.matmul(token_attn, x) # (32, 1, 128)

        if self.n_unsup_concepts > 0:  # unsupervised stream
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_pooled, self.unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            out_n, concept_attn = self.concept_tranformer(x_pooled, self.concepts)
            # out_n.shape: (32, 1, 2), concept_attn.shape: (32, 1, 10)
            concept_attn = concept_attn.mean(1)  # average over heads, concept_attn.shape: (32, 1, 10)
            out = out + out_n.squeeze(1)  # squeeze token dimension, out.shape: (32, 2)

        if self.n_spatial_concepts > 0:  # Spatial stream
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, self.spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


# Shared methods
def ent_loss(probs):
    """Entropy loss"""
    ent = -probs * torch.log(probs + 1e-8)
    return ent.mean()


def concepts_sparsity_cost(concept_attn, spatial_concept_attn):
    '''
    Inputs:
    - concept_attn.shape: (32, 1, 13)
    - spatial_concept_attn.shape: (32, 196, 95)
    '''
    cost = ent_loss(concept_attn) if concept_attn is not None else 0.0
    if spatial_concept_attn is not None:
        cost = cost + ent_loss(spatial_concept_attn)
    return cost


def concepts_cost(concept_attn, attn_targets):
    """Non-spatial concepts cost
        Attention targets are normalized to sum to 1,
        but cost is normalized to be O(1)

    Args:
        attn_targets, torch.tensor of size (batch_size, n_concepts): one-hot attention targets
    
    Inputs:
        CUB:
          concept_attn.shape: (32, 1, 13) , values are float lower than 1 but greater than 0
          attn_targets.shape: (32, 13) , values are binary just 0 or 1
        GlomCrawl:
          concept_attn.shape (990, 1, 630) 
          expl.shape (990, 630)
    """
    if concept_attn is None:
        return 0.0
    if attn_targets.dim() < 3:
        attn_targets = attn_targets.unsqueeze(1) # CUB, shape: (32, 1, 13)
    norm = attn_targets.sum(-1, keepdims=True) # CUB, shape: (32, 1, 1), values are the sum of 1s of a sample
    idx = ~torch.isnan(norm).squeeze() # CUB, all is True, shape: (32)
    if not torch.any(idx):
        return 0.0
    # MSE requires both floats to be of the same type
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float() # softmax, shape: (32, 1, 13)
    n_concepts = norm_attn_targets.shape[-1] # 13
    return n_concepts * F.mse_loss(concept_attn[idx], norm_attn_targets, reduction="mean")


def spatial_concepts_cost(spatial_concept_attn, attn_targets):
    """Spatial concepts cost
        Attention targets are normalized to sum to 1

    Args:
        attn_targets, torch.tensor of size (batch_size, n_patches, n_concepts):
            one-hot attention targets

    Note:
        If one patch contains a `np.nan` the whole patch is ignored

    Inputs:
        CUB:
        - spatial_concept_attn: (32, 196, 95)
        - attn_targets: (32, 196, 95)
    """
    if spatial_concept_attn is None:
        return 0.0
    norm = attn_targets.sum(-1, keepdims=True) # (32, 196, 1)
    idx = ~torch.isnan(norm).squeeze() # False, True, shape: (32, 196), torch.sum(idx) = 353
    if not torch.any(idx):
        return 0.0
    # attn_targets[idx].shape: (353, 95), norm[idx].shape: (353, 1)
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float() # softmax, shape: (353, 95)
    n_concepts = norm_attn_targets.shape[-1] # 95
    return n_concepts * F.mse_loss(spatial_concept_attn[idx], norm_attn_targets, reduction="mean")


class CTC(nn.Module):
    """Concept Transformer Classifier"""

    def __init__(
        self,
        img_size=224,
        embedding_dim=768,
        n_input_channels=3,
        n_conv_layers=1,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_concepts=0,
        n_spatial_concepts=0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.n_concepts = n_concepts # 10
        self.n_spatial_concepts = n_spatial_concepts # 0

        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels, # 1
            n_output_channels=embedding_dim, # 128
            kernel_size=kernel_size, # 3
            stride=stride, # 1
            padding=padding, # 1
            pooling_kernel_size=pooling_kernel_size, # 3
            pooling_stride=pooling_stride, # 2
            pooling_padding=pooling_padding, # 1
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers, # 1
            conv_bias=False,
        )

        sequence_length = self.tokenizer.sequence_length(
            n_channels=n_input_channels, height=img_size, width=img_size
        ) # n_input_channels = 1, img_size = 28, sequence_length = 128
        self.transformer_classifier = TransformerLayer(
            embedding_dim=embedding_dim, sequence_length=sequence_length, *args, **kwargs
        ) # embedding_dim = 128, sequence_length = 196
        self.norm = nn.LayerNorm(embedding_dim) # embedding_dim = 128

        self.concept_transformer = ConceptTransformer(
            embedding_dim=embedding_dim, # 128
            attention_dropout=0.1, 
            projection_dropout=0.1,
            n_concepts=n_concepts, # 10
            n_spatial_concepts=n_spatial_concepts, # 0
            *args,
            **kwargs, # {'num_layers': 2, 'num_heads': 2, 'mlp_ratio': 1, 'num_classes': 2, 'n_unsup_concepts': 0, 'batch_size': 32, 'n_train_samples': 400, 'data_name': 'ExplanationMNIST'}
        )

    def forward(self, x):
        '''
        Inputs:
        - x.shape: (32, 1, 28, 28)

        Outputs:
        - out.shape: (32, 2)
        - unsup_concept_attn.shape: None
        - concept_attn.shape: (32, 1, 10)
        - spatial_concept_attn.
        '''
        x = self.tokenizer(x) # x.shape: (32, 196, 128)
        x = self.transformer_classifier(x) # x.shape: (32, 196, 128)
        x = self.norm(x) # x.shape: (32, 196, 128)
        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.concept_transformer(x)

        return out.squeeze(-2), unsup_concept_attn, concept_attn, spatial_concept_attn


class CTC_ADAPT(nn.Module):
    """Concept Transformer Classifier with pretrained tokenizer and adaptive number of tokens

    Args:
        max_seqlen (int): Maximal sequence length per dimension, which means that if max_seqlen=20 and
            inputs are 2D (images) then the number of tokens is 20*20=400
    """

    def __init__(
        self, embedding_dim=768, n_concepts=0, n_spatial_concepts=0, max_seqlen=14, *args, **kwargs
    ):
        super().__init__()

        self.n_concepts = n_concepts
        self.n_spatial_concepts = n_spatial_concepts

        self.tokenizer = PretrainedTokenizer(
            embedding_dim=embedding_dim, flatten=False, *args, **kwargs
        )
        self.position_embedding = FactorizedPositionEncoding(max_seqlen, 2, embedding_dim)
        self.flattener = nn.Flatten(1, 2)
        self.transformer_classifier = TransformerLayer(
            embedding_dim=embedding_dim, positional_embedding="none", *args, **kwargs
        )
        self.norm = nn.LayerNorm(embedding_dim)

        self.concept_transformer = ConceptTransformer(
            embedding_dim=embedding_dim,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.position_embedding(x)
        x = self.flattener(x)
        x = self.transformer_classifier(x)
        x = self.norm(x)
        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.concept_transformer(x)
        return out.squeeze(-2), unsup_concept_attn, concept_attn, spatial_concept_attn
