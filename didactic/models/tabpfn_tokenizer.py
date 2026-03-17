import contextlib
import itertools
import logging
import os
import random
from abc import ABC
from argparse import Namespace
from pathlib import Path
from shutil import copy2
from typing import Tuple, Union, List, Dict, Optional

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
from tabpfn import TabPFNClassifier # type: ignore[import-untyped]
from tabpfn.inference import InferenceEngineCachePreprocessing # type: ignore[import-untyped]
from tabpfn.architectures.base import PerFeatureTransformer, get_encoder, get_y_encoder # type: ignore[import-untyped]
from tabpfn.model_loading import load_model_criterion_config # type: ignore[import-untyped]
# from tabpfn.utils import _get_ordinal_encoder, _process_text_na_dataframe # type: ignore[import-untyped]
from torch import Tensor


class TabPFNTokenizer(nn.Module, ABC):
    def __init__(self, d_token: int, *args, **kwargs):
        super().__init__()
        self.d_token = d_token

    def _prepare_data_subset(self, tabular_attrs, tabular_num_attrs, tabular_cat_attrs) -> Tuple[Tensor, Tensor | Tuple[Tensor, Tensor]]:
        """Extract and process from the data module, specifically to handle missing values and categorical attributes.

        Args:
            data: ORCHID data module.

        Returns:
            Tuple of data extracted from the subset:
                - DataFrame of tabular data to use as input features, w/ missing values marked as np.nan.
        """

        num_attrs, cat_attrs = None, None
        if tabular_num_attrs:
            # Group the numerical attributes from the `tabular_attrs` input in a single tensor
            num_attrs = torch.hstack([tabular_attrs[attr].unsqueeze(1) for attr in tabular_num_attrs])  # (N, S_num)
        if tabular_cat_attrs:
            # Group the categorical attributes from the `tabular_attrs` input in a single tensor
            cat_attrs = torch.hstack([tabular_attrs[attr].unsqueeze(1) for attr in tabular_cat_attrs])  # (N, S_cat)

        x_num = torch.nan_to_num(num_attrs) if num_attrs is not None else None
        # Keep NaN values for numerical attributes as TabPFNEncoder handles them
        x_cat = cat_attrs.clip(0) if cat_attrs is not None else None

        tab_data = (
            torch.cat([x_num, x_cat], dim=1)
            if x_num is not None and x_cat is not None
            else (x_num if x_num is not None else x_cat)
        )
        tab_notna_mask: Tensor | Tuple[Tensor, Tensor]

        if tab_data is None:
            raise ValueError("At least one of tabular_num_attrs or tabular_cat_attrs must be non-empty")
        if num_attrs is not None and cat_attrs is not None:
            tab_num_notna_mask = ~(num_attrs.isnan())
            tab_cat_notna_mask = cat_attrs != MISSING_CAT_ATTR
            tab_notna_mask = (tab_num_notna_mask, tab_cat_notna_mask)
        elif num_attrs is not None:
            tab_num_notna_mask = ~(num_attrs.isnan())
            tab_notna_mask = tab_num_notna_mask
        elif cat_attrs is not None:
            tab_cat_notna_mask = cat_attrs != MISSING_CAT_ATTR
            tab_notna_mask = tab_cat_notna_mask
        else:
            raise ValueError("At least one of tabular_num_attrs or tabular_cat_attrs must be non-empty")

        return tab_data, tab_notna_mask

    def forward(self, tabular_attrs: PatientData, *args, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        """Tokenize the tabular attributes of a patient.

        Args:
            tabular_attrs: Dictionary of tabular attributes for a patient.

        Returns:
            Tuple of tensors:
                - Tokenized tabular data.
                - Mask indicating which values are not missing.
        """
        # Assert numerical and categorical attributes are provided
        tabular_num_attrs = [attr for attr in tabular_attrs if attr in TabularAttribute.numerical_attrs()]
        tabular_cat_attrs = [attr for attr in tabular_attrs if attr in TabularAttribute.categorical_attrs()]

        # Prepare the data subset
        tab_data, notna_mask = self._prepare_data_subset(tabular_attrs, tabular_num_attrs, tabular_cat_attrs)

        # Artificially expand the tabular data to match the expected input shape for concatenation in the pipeline
        if tab_data.ndim == 1:
            tab_data = tab_data.unsqueeze(1)
        if tab_data.ndim == 2:
            tab_data = einops.repeat(tab_data, "b f -> b f d", d=self.d_token)
        else:
            raise ValueError(f"Unexpected tabular data shape: {tab_data.shape}. Expected 1D or 2D tensor.")

        # No mask for tabular data, as it is already processed to handle missing values in TabPFNEncoder
        notna_mask = torch.full(tab_data.shape[:2], True, device=tab_data.device)

        return tab_data, [notna_mask]
