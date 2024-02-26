import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import sqrt
from einops import rearrange

def TriangularCasualMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.tril(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
        
        @property
        def mask(self):
            return self.mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCasualMask(B, L, device=queries.device)
            
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
# class Cross_Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim ** -0.5

        
#     def forward(self, queries, keys, values, attn_mask=None):
#         q, k, v = map(lambda t: rearrange)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys*n_heads)
        self.key_projection = nn.Linear(d_model, d_keys*n_heads)
        self.value_projection = nn.Linear(d_model, d_values*n_heads)
        self.out_projeciton = nn.Linear(d_values*n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
    
        return self.out_projeciton(out), attn