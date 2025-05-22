import typing as ty
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lib.temporal_embeddings import FullTemporalEmbeddings
from model.lib.temporal_modulation import TemporalModulation


class MLP_Modulated(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        t_mean: float,
        t_std: float,
        d_layers: ty.List[int],
        dropout: float,
        bn: bool,
        transform: bool,
        temporal_embeddings: Optional[dict],
    ) -> None:
        super(MLP_Modulated, self).__init__()
        self.dropout = dropout
        self.bn = bn
        self.transform = transform
        self.temporal_embeddings = FullTemporalEmbeddings(t_mean, t_std, **temporal_embeddings)
        self.d_in = d_in
        self.d_out = d_out
        self.layers = nn.ModuleList([
            nn.Linear(d_layers[i - 1] if i else self.d_in, x)
            for i, x in enumerate(d_layers)
        ])
        self.head = nn.Linear(d_layers[-1] if d_layers else self.d_in, self.d_out)
        self.modulator_layers = nn.ModuleList([
            TemporalModulation(self.temporal_embeddings.out_dim, x, transform) for x in d_layers
        ])
        self.modulator_in = TemporalModulation(self.temporal_embeddings.out_dim, self.d_in, transform)
        self.modulator_out = TemporalModulation(self.temporal_embeddings.out_dim, self.d_out, transform)
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(x) for x in d_layers
        ])

    def forward(self, x, x_cat, idx):
        if self.temporal_embeddings.out_dim:
            idx = self.temporal_embeddings(idx)

        x = self.modulator_in(x, idx)
        
        for layer, bn_layer, modulator_layer in zip(self.layers, self.bn_layers, self.modulator_layers):
            x = layer(x)
            if self.bn:
                x = bn_layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
            x = modulator_layer(x, idx)
        
        logit = self.head(x)
        logit = self.modulator_out(logit, idx)
        
        if self.d_out == 1:
            logit = logit.squeeze(-1)
        return logit