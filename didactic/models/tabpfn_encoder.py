import contextlib
import itertools
import logging
import os
import random
from functools import partial
from abc import ABC
from argparse import Namespace
from pathlib import Path
from shutil import copy2
from typing import Tuple, Union, Iterable, Optional, Literal

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataprocessing.data.config import Subset
from dataprocessing.data.orchid.config import OrchidTag, TabularAttribute

# from dataprocessing.data.orchid.data_module import OrchidDataModule
from dataprocessing.data.orchid.datapipes import MISSING_CAT_ATTR, PatientData, filter_time_series_attributes
from pytorch_lightning.trainer.states import TrainerFn
from tabpfn import TabPFNClassifier  # type: ignore[import-untyped]
from tabpfn.inference import InferenceEngineCachePreprocessing  # type: ignore[import-untyped]
from tabpfn.architectures.base import PerFeatureTransformer, get_encoder, get_y_encoder  # type: ignore[import-untyped]
from tabpfn.model_loading import load_model_criterion_config
# from didactic.models.model_utils import rope_compute_heads_wrapper
from didactic.models.model_utils import rope_compute_heads_wrapper
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention


class TabPFNEncoder(nn.Module, ABC):
    # def __init__(self, modelPFN: PerFeatureTransformer, state_dict_path: str, d_model: int, seed: int = 42, **kwargs):
    def __init__(
        self,
        model_path: str,
        which: str,
        fit_mode: str,
        seed: int,
        n_time_series_attrs: int,
        features_per_group: Optional[int] = 1,
        updated_pfn_path: Union[Path, None] = None,
        random_init: bool = False,
        recompute_layer: bool = False,
        num_channels: int = 4,
        time_points: int = 64*4,
        positional_encoding: Literal["none", "sinusoidal", "rope", "learned"] = "none",
        **kwargs,
    ):
        super().__init__()
        model_list, _, self.model_config, _ = load_model_criterion_config(
            model_path=model_path,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=(fit_mode == "fit_with_cache"),
            which="classifier",
            version="v2",
            download_if_not_exists=False,
            # download=False,
        )
        self.recompute_layer = recompute_layer
        self.model = model_list[0]

        if updated_pfn_path is not None:
            # Load updated model weights after pretraining
            logging.info(f"Loading updated TabPFN model weights from {updated_pfn_path}")
            state_dict = torch.load(updated_pfn_path, map_location="cuda:0")  # updated_pfn_path is already a state dict
            if positional_encoding == "learned" and "pe" in state_dict:
                pe_state = state_dict.pop("pe")
                with torch.no_grad():
                    self.pe.copy_(pe_state)
            if "channel_positional_encoding" in state_dict:
                channel_pe_state = state_dict.pop("channel_positional_encoding")
                # with torch.no_grad():
                #     self.channel_positional_encoding.copy_(channel_pe_state)
            self.model.load_state_dict(state_dict, strict=True)

        if random_init:  # random_init:
            self.model.apply(self._init_weights)
            logging.info("Randomly initialized TabPFN model weights")
        else:
            logging.info("Loaded pretrained TabPFN model weights")

        # self.ninp = ninp
        # self.random_embedding_seed = seed
        self.encoder = self.model.encoder
        self.y_encoder = self.model.y_encoder
        self.transformer_encoder = self.model.transformer_encoder
        self.features_per_group = (
            self.model.features_per_group if features_per_group is None else features_per_group
        )
        self.n_time_series_attrs = n_time_series_attrs
        self.positional_encoding = positional_encoding
        self.num_channels = num_channels
        self.time_points = time_points
        
        if self.positional_encoding == "rope":
            # Modify attention mechanism to include RoPE with channel-wise application
            original_static_compute = MultiHeadAttention.compute_attention_heads
            patched_rope = partial(
                rope_compute_heads_wrapper,
                original_func=original_static_compute,
                num_channels=self.num_channels,
                time_points=self.time_points,
            )
            MultiHeadAttention.compute_attention_heads = staticmethod(patched_rope)
            for layer in self.transformer_encoder.layers:
                layer.self_attn_between_features.is_feature_attn = True
                layer.self_attn_between_items.is_feature_attn = False
        
        # self.feature_positional_embedding_embeddings = nn.Linear(self.ninp // 4, self.ninp)

    def _init_weights(self, module):
        # Initialize Linear layers (with or without bias)
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize LayerNorm layers (weight to 1, bias to 0)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize embeddings using normal distribution if any
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=1)

    def sinusoidal_positional_encoding(self, sequence_length=64, embedding_dim=192, n=10000.0):
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even")

        positions = torch.arange(0, sequence_length).unsqueeze(1)  # Shape: (sequence_length, 1)
        denominators = torch.pow(
            n, 2 * torch.arange(0, embedding_dim // 2) / embedding_dim
        )  # Shape: (embedding_dim/2,)

        posenc = torch.zeros(sequence_length, embedding_dim)
        posenc[:, 0::2] = torch.sin(positions / denominators)  # Apply sin to even indices
        posenc[:, 1::2] = torch.cos(positions / denominators)  # Apply cos to odd indices

        return posenc

    def learnable_positional_encoding(self, sequence_length=64, embedding_dim=192):
        # Use nn.Parameter
        pos_enc = nn.Parameter(torch.zeros(sequence_length, embedding_dim))
        nn.init.normal_(pos_enc, mean=0, std=0.02)
        return pos_enc

    def encode_x_and_y(
        self,
        X: Tensor,
        y: Tensor,
    ) -> Tuple[Tensor, Tensor, int]:
        single_eval_pos_ = y.shape[0]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.ndim == 2:
            y = y.unsqueeze(-1)  # (S, B) -> (S, B, 1)

        y = y.transpose(0, 1)  # (B, S, 1)

        assert y.shape[1] == single_eval_pos_

        X = einops.rearrange(X, "s b (f n) -> b s f n", n=self.features_per_group)
        y = torch.cat(
            (
                y,
                torch.nan
                * torch.zeros(
                    y.shape[0],
                    X.shape[1] - y.shape[1],
                    y.shape[2],
                    device=y.device,
                    dtype=y.dtype,
                ),
            ),
            dim=1,
        )

        y = y.transpose(0, 1)  # (Seq, N, 1)
        y[single_eval_pos_:] = torch.nan  # Make sure that no label leakage ever happens

        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=False,
        ).transpose(0, 1)

        assert not torch.isnan(embedded_y).any(), f"{torch.isnan(embedded_y).any()=}, Make sure to add nan handlers"

        X = einops.rearrange(X, "b s f n -> s (b f) n")
        embedded_x = einops.rearrange(
            self.encoder(
                {"main": X},
                single_eval_pos=single_eval_pos_,
                cache_trainset_representation=False,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )
        return embedded_x, embedded_y, single_eval_pos_

    def forward(
        self, X_full: Tensor, y_train: Tensor, use_ts_sinus: str = "none", *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        seq_len, batch_size, num_features = X_full.shape

        emb_x, emb_y, single_eval_pos = self.encode_x_and_y(X_full, y_train)

        if use_ts_sinus == "sinusoidal":
            # Add sinusoidal positional encodings to time series attributes
            pos_enc = self.sinusoidal_positional_encoding().to(emb_x.device)  # (T, E)
            # Multiply pos_enc sequence by number of time series attributes
            pos_enc = pos_enc.unsqueeze(1).repeat(1, self.n_time_series_attrs, 1)  # (T, F_ts, E)
            pos_enc = pos_enc.view(64 * self.n_time_series_attrs, -1).unsqueeze(0)  # (1, T * F_ts, E)
            emb_x += pos_enc.unsqueeze(0)  # (1, 1, T * F_ts, E) -> broadcast to (1, S, T * F_ts, E)
        elif use_ts_sinus == "learned":
            # Add learnable positional encodings to time series attributes
            emb_x, emb_y = self.model.add_embeddings(
                emb_x,
                emb_y,
                data_dags=None,
                num_features=num_features,
                seq_len=seq_len,
            )
            pos_enc = self.learnable_positional_encoding().to(emb_x.device)  # (T, E)
            # Multiply pos_enc sequence by number of time series attributes
            pos_enc = pos_enc.unsqueeze(1).repeat(1, self.n_time_series_attrs, 1)  # (T, F_ts, E)
            pos_enc = pos_enc.view(64 * self.n_time_series_attrs, -1).unsqueeze(0)  # (1, T * F_ts, E)
            emb_x += pos_enc.unsqueeze(0)  # (1, 1, T * F_ts, E) -> broadcast to (1, S, T * F_ts, E)
        elif use_ts_sinus == "none":
            emb_x, emb_y = self.model.add_embeddings(
                emb_x,
                emb_y,
                data_dags=None,
                num_features=num_features,
                seq_len=seq_len,
            )

        # (B, S, F, E) + (B, S, 1, E) -> (B, S, F + 1, E)
        embedded_input = torch.cat((emb_x, emb_y.unsqueeze(2)), dim=2)
        assert not torch.isnan(
            embedded_input
        ).any(), f"{torch.isnan(embedded_input).any()=}, Make sure to add nan handlers"

        output = self.transformer_encoder(
            embedded_input,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=False,
            recompute_layer=self.recompute_layer,
            # half_layers=False,
            save_peak_mem_factor=None,
        )
        out_query = output[:, single_eval_pos:, :].transpose(0, 1)
        query_encoder_out = out_query.squeeze(1)  # (S_query, B=1, F, E) -> (S_query, F, E)

        return query_encoder_out
