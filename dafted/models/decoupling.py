import math
from typing import Callable, Dict, Literal, Tuple, Union

import hydra
import torch
from omegaconf import DictConfig
from scipy.special import binom, factorial
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init

import dafted.models.transformer
from dafted.models.layers import _QKVLinearProjection, _QKVMatrixMultiplication

from .layers import get_nn_module

ModuleType = Union[str, Callable[..., nn.Module]]


class DecouplingModule(nn.Module):
    decoupling: nn.Module

    def __init__(
        self,
        ts_input_size: int,
        ts_proj_size: int,
        tab_input_size: int,
        tab_proj_size: int,
        decoupling_method: str = "linear",
        **kwargs,
    ):
        super().__init__()
        if decoupling_method == "linear":
            self.decoupling = LinearDecoupling(
                tab_input_size=tab_input_size,
                tab_proj_size=tab_proj_size,
                ts_input_size=ts_input_size,
                ts_proj_size=ts_proj_size,
            )
        elif decoupling_method == "mlp":
            self.decoupling = MLPDecoupling(
                tab_input_size=tab_input_size,
                tab_proj_size=tab_proj_size,
                ts_input_size=ts_input_size,
                ts_proj_size=ts_proj_size,
            )
        elif decoupling_method == "big_mlp":
            self.decoupling = BigMLPDecoupling(
                tab_input_size=tab_input_size,
                tab_proj_size=tab_proj_size,
                ts_input_size=ts_input_size,
                ts_proj_size=ts_proj_size,
            )

    def forward(self, ts_tokens: Tensor, tab_tokens: Tensor) -> Tensor:
        return self.decoupling(ts_tokens, tab_tokens)


class LinearDecoupling(nn.Module):
    def __init__(
        self,
        ts_input_size: int,
        ts_proj_size: int,
        tab_input_size: int,
        tab_proj_size: int,
    ):
        super().__init__()
        self.tab_split = int(tab_proj_size / 2)
        self.linear_ts = nn.Linear(ts_input_size, ts_proj_size)
        self.linear_tab = nn.Linear(tab_input_size, tab_proj_size)

    def forward(self, ts_tokens: Tensor, tab_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ts_tokens = self.linear_ts(ts_tokens)
        tab_tokens = self.linear_tab(tab_tokens)
        tab_tokens_specific = tab_tokens.reshape(tab_tokens.shape[0], -1, self.tab_split)[:, ::2, :]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.tab_split)[:, 1::2, :]
        return ts_tokens, tab_tokens_specific, tab_tokens_shared


class MLPDecoupling(nn.Module):
    def __init__(
        self,
        tab_input_size: int,
        tab_proj_size: int,
        ts_input_size: int,
        ts_proj_size: int,
    ):
        super().__init__()
        self.tab_split = int(tab_proj_size / 2)
        self.mlp_ts = nn.Sequential(
            nn.Linear(ts_input_size, ts_proj_size),
            nn.GELU(),
            nn.Linear(ts_proj_size, ts_proj_size),
        )
        self.mlp_tab = nn.Sequential(
            nn.Linear(tab_input_size, tab_proj_size),
            nn.GELU(),
            nn.Linear(tab_proj_size, tab_proj_size),
        )

    def forward(self, ts_tokens: Tensor, tab_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ts_tokens = self.mlp_ts(ts_tokens)
        tab_tokens = self.mlp_tab(tab_tokens)
        tab_tokens_specific = tab_tokens.reshape(tab_tokens.shape[0], -1, self.tab_split)[:, ::2, :]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.tab_split)[:, 1::2, :]
        return ts_tokens, tab_tokens_specific, tab_tokens_shared


class BigMLPDecoupling(nn.Module):
    def __init__(
        self,
        tab_input_size: int,
        tab_proj_size: int,
        ts_input_size: int,
        ts_proj_size: int,
    ):
        super().__init__()
        self.tab_split = int(tab_proj_size / 2)
        self.mlp_ts = nn.Sequential(
            nn.Linear(ts_input_size, ts_proj_size),
            nn.GELU(),
            nn.Linear(ts_proj_size, ts_proj_size),
            nn.GELU(),
            nn.Linear(ts_proj_size, ts_proj_size),
        )
        self.mlp_tab = nn.Sequential(
            nn.Linear(tab_input_size, tab_proj_size),
            nn.GELU(),
            nn.Linear(tab_proj_size, tab_proj_size),
            nn.GELU(),
            nn.Linear(tab_proj_size, tab_proj_size),
        )

    def forward(self, ts_tokens: Tensor, tab_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ts_tokens = self.mlp_ts(ts_tokens)
        tab_tokens = self.mlp_tab(tab_tokens)
        tab_tokens_specific = tab_tokens.reshape(tab_tokens.shape[0], -1, self.tab_split)[:, ::2, :]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.tab_split)[:, 1::2, :]
        return ts_tokens, tab_tokens_specific, tab_tokens_shared
