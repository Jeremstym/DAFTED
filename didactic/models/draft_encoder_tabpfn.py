import contextlib
import itertools
import logging
import os
import random
from abc import ABC
from argparse import Namespace
from pathlib import Path
from shutil import copy2
from typing import Tuple, Union

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataprocessing.data.config import Subset
from dataprocessing.data.orchid.config import OrchidTag, TabularAttribute
from dataprocessing.data.orchid.data_module import OrchidDataModule
from dataprocessing.data.orchid.datapipes import MISSING_CAT_ATTR, PatientData, filter_time_series_attributes
from pytorch_lightning.trainer.states import TrainerFn
from tabpfn import TabPFNClassifier  # type: ignore[import-untyped]
from tabpfn.inference import InferenceEngineCachePreprocessing  # type: ignore[import-untyped]
from tabpfn.architectures.base import PerFeatureTransformer, get_encoder, get_y_encoder  # type: ignore[import-untyped]
from tabpfn.model_loading import load_model_criterion_config

# from tabpfn.utils import _get_ordinal_encoder, _process_text_na_dataframe  # type: ignore[import-untyped]
from torch import Tensor


class TabPFNEncoder(nn.Module, ABC):
    # def __init__(self, modelPFN: PerFeatureTransformer, state_dict_path: str, d_model: int, seed: int = 42, **kwargs):
    def __init__(self, model_path: str, which: str, fit_mode: str, seed: int, inner_prediction_head: bool = False, **kwargs):
        super().__init__()
        # self.device = device
        # self.seed = seed
        # self.model = modelPFN(
        #     seed=seed,
        #     ninp=192,
        #     nhead=6,
        #     nhid=768,
        #     feature_positional_embedding="subspace",
        #     nlayers=12,
        #     encoder=get_encoder(
        #         num_features=1,
        #         embedding_size=192,
        #         remove_empty_features=True,
        #         remove_duplicate_features=False,
        #         nan_handling_enabled=True,
        #         normalize_on_train_only=True,
        #         normalize_to_ranking=False,
        #         normalize_x=True,
        #         remove_outliers=False,
        #         normalize_by_used_features=True,
        #         encoder_use_bias=False,
        #     ),
        #     y_encoder=get_y_encoder(
        #         num_inputs=1,
        #         embedding_size=192,
        #         nan_handling_y_encoder=True,
        #         max_num_classes=10,
        #     ),
        #     multiquery_item_attention_for_test_set=False,
        #     decoder_dict={"standard": (None, 10)},
        # )
        self.model, _, self.model_config = load_model_criterion_config(
            model_path=model_path,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=(fit_mode == "fit_with_cache"),
            which="classifier",
            version="v2",
            download=False,
        )

        # self.model.load_state_dict(model_ckpt.state_dict(), strict=True)

        self.x_encoder = self.model.encoder
        self.y_encoder = self.model.y_encoder
        self.transformer_encoder = self.model.transformer_encoder

        self.inner_prediction_head = inner_prediction_head
        if self.inner_prediction_head:
            self.decoder_dict = self.model.decoder_dict

    # def _prepare_data_subset(self, tabular_attrs, tabular_num_attrs, tabular_cat_attrs) -> Tuple[pd.DataFrame, np.ndarray]:
    #     """Extract and process from the data module, specifically to handle missing values and categorical attributes.

    #     Args:
    #         data: ORCHID data module.

    #     Returns:
    #         Tuple of data extracted from the subset:
    #             - DataFrame of tabular data to use as input features, w/ missing values marked as np.nan.
    #     """

    #     # For each tabular feature, save the vectors of values over each batch
    #     # tab_data = {}
    #     # for batch, attr in tabular_attrs:
    #     #     attr_batch_data = batch[attr].detach().cpu().numpy()
    #     #     tab_data.setdefault(attr, []).append(attr_batch_data)

    #     # Concatenate the vectors of batches of tabular features into vectors over the entire training set
    #     # tab_data = {tab_attr: np.hstack(batch_vals) for tab_attr, batch_vals in tab_data.items()}

    #     # tab_data = {attr.__str__(): tabular_attrs[attr] for attr in tabular_attrs}

    #     # # Create a dataframe for the training data,
    #     # # and cast categorical attributes to the appropriate data type
    #     # tab_tags = tuple(TabularAttribute[e] for e in tabular_attrs)
    #     # cat_attrs = [attr for attr in tab_tags if attr in TabularAttribute.categorical_attrs()]
    #     # tab_df = pd.DataFrame(tab_data).astype({attr: "category" for attr in cat_attrs})

    #     # # After casting categorical attributes to the appropriate data type,
    #     # # mark missing values as `np.nan` so that they can be handled properly by the model
    #     # tab_df[cat_attrs] = tab_df[cat_attrs].replace(MISSING_CAT_ATTR, np.nan)

    #     num_attrs, cat_attrs = None, None
    #     if tabular_num_attrs:
    #         # Group the numerical attributes from the `tabular_attrs` input in a single tensor
    #         num_attrs = torch.hstack(
    #             [tabular_attrs[attr].unsqueeze(1) for attr in tabular_num_attrs]
    #         )  # (N, S_num)
    #     if tabular_cat_attrs:
    #         # Group the categorical attributes from the `tabular_attrs` input in a single tensor
    #         cat_attrs = torch.hstack(
    #             [tabular_attrs[attr].unsqueeze(1) for attr in tabular_cat_attrs]
    #         )  # (N, S_cat)

    #     x_num = torch.nan_to_num(num_attrs) if num_attrs is not None else None
    #     x_cat = cat_attrs.clip(0) if cat_attrs is not None else None

    #     tab_data = torch.cat(
    #         [x_num, x_cat], dim=1
    #     ) if x_num is not None and x_cat is not None else (
    #         x_num if x_num is not None else x_cat
    #     )
    #     if tab_data is None:
    #         raise ValueError(
    #             "At least one of tabular_num_attrs or tabular_cat_attrs must be non-empty"
    #         )
    #     if tabular_num_attrs and tabular_cat_attrs:
    #         tab_num_notna_mask = ~(num_attrs.isnan())
    #         tab_cat_notna_mask = (cat_attrs != MISSING_CAT_ATTR)
    #         tab_notna_mask = (tab_num_notna_mask, tab_cat_notna_mask)
    #     elif tabular_num_attrs:
    #         tab_num_notna_mask = ~(num_attrs.isnan())
    #         tab_notna_mask = tab_num_notna_mask
    #     elif tabular_cat_attrs:
    #         tab_cat_notna_mask = (cat_attrs != MISSING_CAT_ATTR)
    #         tab_notna_mask = tab_cat_notna_mask
    #     else:
    #         raise ValueError(
    #             "At least one of tabular_num_attrs or tabular_cat_attrs must be non-empty"
    #         )

    #     return tab_data, tab_notna_mask

    def encode_x_and_y(
        self,
        X: Tensor,
        y: Tensor,
    ) -> Tuple[Tensor, Tensor, int]:
        # x = torch.as_tensor(X[:, :-1], dtype=torch.float32).unsqueeze(1)
        # y = torch.as_tensor(X[:,-1], dtype=torch.float32).unsqueeze(1)
        single_eval_pos_ = y.shape[0]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.ndim == 2:
            y = y.unsqueeze(-1)  # (Seq, N) -> (Seq, N, 1)

        y = y.transpose(0, 1)  # (N, Seq, 1)

        assert y.shape[1] == single_eval_pos_
        
        X = einops.rearrange(X, "s b (f n) -> b s f n", n=self.model.features_per_group)
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

        # with contextlib.suppress(EException):
        #     X = X.float()
        X = einops.rearrange(X, "b s f n -> s (b f) n")
        embedded_x = einops.rearrange(
            self.x_encoder(
                {"main": X},
                single_eval_pos=single_eval_pos_,
                cache_trainset_representation=False,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )
        return embedded_x, embedded_y, single_eval_pos_

    def forward(self, X_full: Tensor, y_train: Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Retrive the tabular data, has it is no token yet
        # X = X_full[:, :-1, 0]  # The values are repeated across the d_token dimension, so we can take the first one
        # y = y_train[:, -1].mean(dim=1, keepdim=True)  # The last feature is the target variable

        seq_len, batch_size, num_features = X_full.shape

        emb_x, emb_y, single_eval_pos = self.encode_x_and_y(X_full, y_train)
        emb_x, emb_y = self.model.add_embeddings(
            emb_x,
            emb_y,
            data_dags=None,
            num_features=num_features,
            seq_len=seq_len,
        )

        # (N, Seq, num_features, d_model) + (N, Seq, 1, d_model) -> (N, Seq, num_features + 1, d_model)
        embedded_input = torch.cat((emb_x, emb_y.unsqueeze(2)), dim=2)
        assert not torch.isnan(
            embedded_input
        ).any(), f"{torch.isnan(embedded_input).any()=}, Make sure to add nan handlers"

        output = self.transformer_encoder(
            embedded_input,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=False,
            # half_layers=False,
        )
        out_query = output[:, single_eval_pos:, -1].transpose(0, 1)  # (N_query, d_model)
        # train_encoder_out = output[:, :, -1].transpose(0, 1)
        # train_encoder_out = output[:, :, :-1].transpose(0, 1).squeeze(1)
        if self.inner_prediction_head:
            # If the inner prediction head is used, apply it to the output of the transformer encoder
            # to get the predictions for the training data points
            assert "standard" in self.decoder_dict
            output = self.decoder_dict["standard"](out_query)
            return output
        else:
            query_encoder_out = out_query.squeeze(1)
            # assert query_encoder_out.shape == (tab_data.shape[0], 1, self.d_model), \
            #     f"Expected output shape {(tab_data.shape[0], 1, self.d_model)}, got {query_encoder_out.shape}"

            return query_encoder_out
