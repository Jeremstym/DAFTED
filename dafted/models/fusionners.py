from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F


class MLPFusion(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        n_layers: int = 2,
        **kwargs
    ):
        super(MLPFusion, self).__init__()
        prev_dim = 2 * emb_dim
        hidden_dim = hidden_dim if hidden_dim is not None else emb_dim

        layers = []
        # First layer: input to hidden
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Middle layers: hidden to hidden
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Last layer: hidden to output
        layers.append(nn.Linear(hidden_dim, emb_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)
