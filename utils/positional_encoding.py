import torch
import torch.nn as nn
import numpy as np

class SineCosPE(nn.Module):
    def __init__(self, input_dim, N_freqs=16, max_freq=4, periodic_fns=[torch.sin, torch.cos],
                 log_sampling=True, include_input=True, trainable=False):
        super().__init__()

        self.periodic_fns = periodic_fns
        self.include_input = include_input or len(periodic_fns) == 0
        self.out_dim = len(periodic_fns) * input_dim * N_freqs

        if self.include_input:
            self.out_dim += input_dim
        
        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        if trainable:
            self.freq_bands = nn.Parameter(freq_bands, requires_grad=True)
        else:
            self.register_buffer('freq_bands', freq_bands, persistent=False)
    
    def forward(self, inputs):
        N_freqs = len(self.freq_bands)

        embeds = []
        for periodic_fn in self.periodic_fns:
            x_freq = inputs[..., None].expand(inputs.shape+(N_freqs,)) * self.freq_bands
            x_freq = periodic_fn(x_freq)
            embeds.append(x_freq.transpose(-1, -2))
        
        embeds = torch.stack(embeds, -2)
        embeds = embeds.reshape(inputs.shape[:-1]+(-1,))

        if self.include_input:
            embeds = torch.cat([inputs, embeds], -1)
        
        return embeds


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(256, num_pos_feats)
        self.col_embed = nn.Embedding(256, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        # input: x, [b, N, 2]
        # output: [b, N, C]

        h = w = int(np.sqrt(x.shape[1]))
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1).view(x.shape[0], h * w, -1)
        # print('pos', pos.shape)
        return pos