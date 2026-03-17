import functools
import importlib
import itertools
import logging
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, cast

import hydra
import torch
from dataprocessing.data.augmentation.base import mask_tokens, random_masking
from dataprocessing.data.orchid.config import OrchidTag, TabularAttribute, TimeSeriesAttribute
from dataprocessing.data.orchid.config import View as ViewEnum
from dataprocessing.data.orchid.datapipes import (
    MISSING_CAT_ATTR,
    PatientData,
    PatientDataTarget,
    PatientDataInference,
    filter_time_series_attributes,
)
from dataprocessing.data.orchid.utils.attributes import TABULAR_CAT_ATTR_LABELS
from dataprocessing.tasks.generic import SharedStepsTask
from dataprocessing.utils.decorators import auto_move_data
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig

# import dataprocessing
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict, init
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, auroc, average_precision, f1_score, mean_absolute_error
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryAveragePrecision,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics import MetricCollection

from didactic.utils.compliance import check_model_encoder
from didactic.models.time_series import TimeSeriesFeatureFusion

# from didactic.utils.tabular_preprocessing import preprocess_tensor, X_input_preprocessing
import didactic.models.transformer
from didactic.models.adaptater import (
    AdapterWrapperFT_Interleaved,
    AdapterWrapperFT_Transformer,
    AdapterWrapperFT_Transformer_CrossAtt,
    LoRALinear,
)
from didactic.models.layers import CLSToken, PositionalEncoding, SequencePooling
from didactic.models.losses import ReconstructionLoss
from didactic.models.tabular import TabularEmbedding
from didactic.models.time_series import TimeSeriesEmbedding, ts_collapse_on_features
import didactic.models.fusionners
import didactic.models.transformer

logger = logging.getLogger(__name__)

torch.set_printoptions(threshold=10000)


class CardiacMultimodalTabPFN(SharedStepsTask):
    """Multi-modal transformer to learn a representation from cardiac imaging and patient records data."""

    def __init__(
        self,
        embed_dim: int,
        tabular_attrs: Sequence[TabularAttribute | str],
        time_series_attrs: Sequence[TimeSeriesAttribute],
        target_attr: TabularAttribute | str,
        views: Sequence[ViewEnum] = tuple(ViewEnum),
        predict_losses: Optional[Dict[TabularAttribute | str, Callable[[Tensor, Tensor], Tensor]] | DictConfig] = None,
        tabular_tokenizer: Optional[TabularEmbedding | DictConfig] = None,
        time_series_tokenizer: Optional[TimeSeriesEmbedding | DictConfig] = None,
        time_series_processing_option: Literal["none", "collapse"] = "none",
        time_series_positional_encoding: Literal["none", "sinusoidal", "learned"] = "none",
        time_series_feature_fusion: bool = False,
        mtr_p: float | Tuple[float, ...] = 0.0,
        mt_by_attr: bool = False,
        # perform_lora: bool = False,
        *args,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            embed_dim: Size of the tokens/embedding for all the modalities.
            tabular_attrs: Tabular attributes to provide to the model.
            time_series_attrs: Time-series attributes to provide to the model.
            views: Views from which to include time-series attributes.
            predict_losses: Supervised criteria to measure the error between the predicted attributes and their real
                value.
            tabular_tokenizer: Tokenizer that can process tabular, i.e. patient records, data.
            time_series_tokenizer: Tokenizer that can process time-series data.
            cross_attention_module: Module to use for cross-attention between the tabular and time-series tokens.
            mtr_p: Probability to replace tokens by the learned MASK token, following the Mask Token Replacement (MTR)
                data augmentation method.
                If a float, the value will be used as masking rate during training (disabled during inference).
                If a tuple, specify a masking rate to use during training and inference, respectively.
            mt_by_attr: Whether to use one MASK token per attribute (`True`), or one universal MASK token for all
                attributes (`False`).
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        # Ensure string tags are converted to their appropriate enum types
        # And do it before call to the parent's `init` so that the converted values are saved in `hparams`
        tabular_attrs = tuple(TabularAttribute[e] for e in tabular_attrs)
        views = tuple(ViewEnum[e] for e in views)
        time_series_attrs = tuple(TimeSeriesAttribute[e] for e in time_series_attrs)

        print(f"number of tabular attributes: {len(tabular_attrs)}")
        print(f"number of time series attributes: {len(time_series_attrs)}")
        print(f"number of views: {len(views)}")

        # If dropout/masking are not single numbers, make sure they are tuples (and not another container type)
        if not isinstance(mtr_p, float):
            assert not isinstance(mtr_p, int)
            mtr_p = tuple(mtr_p)
            assert len(mtr_p) == 2

        if time_series_attrs:
            if not time_series_tokenizer:
                raise ValueError(
                    f"You have requested the following time-series attributes: "
                    f"{[str(attr) for attr in time_series_attrs]}, but have not configured a tokenizer for time-series "
                    f"attributes. Either provide this tokenizer (through the `time_series_tokenizer` parameter) or "
                    f"remove any time-series attributes (by setting the `time_series_attrs` to be an empty list)."
                )
            if getattr(time_series_tokenizer, "model") is None:
                logger.warning(
                    f"You have requested the following time-series attributes: "
                    f"{[str(attr) for attr in time_series_attrs]}, but have not configured a model for the tokenizer "
                    f"for time-series attributes. The tokenizer's model is optional, but highly recommended, so this "
                    f"is likely an oversight. You can provide this model through the `time_series_tokenizer.model` "
                    f"parameter."
                )
        if not (tabular_attrs or time_series_attrs):
            raise ValueError(
                "You configured neither tabular attributes nor time-series attributes as input variables to the model, "
                "but the model requires at least one input. Set non-empty values for either or both `tabular_attrs` "
                "and `time_series_attrs`."
            )

        super().__init__(*args, **kwargs)

        if self.hparams["model"].encoder.get("n_bidirectional_blocks", None) and not (
            tabular_attrs and time_series_attrs
        ):
            raise ValueError(
                "You have configured a multimodal cross-attention module, but either the tabular or the time-series "
                "tabular or the time-series attributes are missing. Make sure to provide both tabular and time-series "
                "attributes when configuring a cross-attention module."
            )
        if self.hparams["model"].encoder.get("n_cross_blocks", None) and not (tabular_attrs and time_series_attrs):
            raise ValueError(
                "You have configured a multimodal cross-attention module, but either the tabular or the time-series "
                "tabular or the time-series attributes are missing. Make sure to provide both tabular and time-series "
                "attributes when configuring a cross-attention module."
            )

        # TOFIX: Hack to log time-series tokenizer model's hparams when it's a config for a `torch.nn.Sequential` object
        # In that case, we have to use a `ListConfig` for the reserved `_args_` key. However, the automatic
        # serialization of `_args_` fails (w/ a '`DictConfig' not JSON serializable' error). Therefore, we fix it by
        # manually unpacking and logging the first and only element in the `_args_` `ListConfig`
        if isinstance(time_series_tokenizer, DictConfig):
            if time_series_tokenizer.get("model", {}).get("_target_") == "torch.nn.Sequential":
                self.save_hyperparameters(
                    {"time_series_tokenizer/model/_args_/0": time_series_tokenizer.model._args_[0]}
                )

        # Add shortcut to lr to work with Lightning's learning rate finder
        self.hparams["lr"] = None

        # Add shortcut to token labels to avoid downstream applications having to determine them from hyperparameters
        self.token_tags = (
            tuple("/".join([view, attr]) for view, attr in itertools.product(views, time_series_attrs)) + tabular_attrs
        )

        # Categorise the tabular attributes in terms of their type (numerical vs categorical)
        self.tabular_num_attrs = [
            attr for attr in self.hparams["tabular_attrs"] if attr in TabularAttribute.numerical_attrs()
        ]
        self.tabular_cat_attrs = [
            attr
            for attr in self.hparams["tabular_attrs"]
            if attr in TabularAttribute.categorical_attrs() and attr != target_attr
        ]
        self.tabular_cat_attrs_cardinalities = [
            len(TABULAR_CAT_ATTR_LABELS[cat_attr]) for cat_attr in self.tabular_cat_attrs
        ]

        self.cat_idxs = []
        for attr in self.tabular_cat_attrs:
            self.cat_idxs.append(self.hparams["tabular_attrs"].index(attr))

        self.target_attr = target_attr

        # self.preprocessing_config = hydra.utils.instantiate(self.hparams["data"]["preprocessors"])

        # Extract train/test masking probabilities from their configs
        if isinstance(self.hparams["mtr_p"], tuple):
            self.train_mtr_p, self.test_mtr_p = self.hparams["mtr_p"]
        else:
            self.train_mtr_p = self.hparams["mtr_p"]
            self.test_mtr_p = 0

        # Configure losses/metrics to compute at each train/val/test step
        self.metrics = nn.ModuleDict()

        # Supervised losses and metrics
        self.predict_losses = {}
        if predict_losses:
            self.predict_losses = {
                TabularAttribute[attr]: (  # type: ignore[misc]
                    hydra.utils.instantiate(attr_loss) if isinstance(attr_loss, DictConfig) else attr_loss
                )
                for attr, attr_loss in predict_losses.items()
            }
        self.hparams["target_tabular_attrs"] = tuple(
            self.predict_losses
        )  # Hyperparameter to easily access target attributes
        for attr in self.predict_losses:
            if attr in TabularAttribute.numerical_attrs():
                self.metrics[attr] = MetricCollection([MeanAbsoluteError(), MeanSquaredError()])
            elif attr in TabularAttribute.binary_attrs():
                self.metrics[attr] = MetricCollection(
                    [
                        BinaryAccuracy(),
                        BinaryAUROC(),
                        BinaryAveragePrecision(),
                        BinaryF1Score(),
                    ],
                )
            else:  # attr in TabularAttribute.categorical_attrs()
                num_classes = len(TABULAR_CAT_ATTR_LABELS[attr])
                self.metrics[attr] = MetricCollection(
                    [
                        MulticlassAccuracy(num_classes=num_classes, average="micro"),
                        MulticlassAUROC(num_classes=num_classes, average="macro"),
                        MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
                        MulticlassF1Score(num_classes=num_classes, average="macro"),
                    ]
                )

        # Compute shapes relevant for defining the models' architectures
        self.n_tabular_attrs = len(self.hparams["tabular_attrs"])
        self.n_time_series_attrs = len(self.hparams["time_series_attrs"]) * len(self.hparams["views"])
        self.sequence_length = self.n_time_series_attrs + self.n_tabular_attrs

        # Self-supervised losses and metrics
        # Initialize transformer encoder and self-supervised + prediction heads
        self.tabular_encoder, self.contrastive_head, self.prediction_heads = self.configure_model()

        self.fusion_module = hydra.utils.instantiate(
            self.hparams["model"].get("fusion_module", None),
        )

        if self.hparams["model"].get("use_secondary_encoder", False):
            self.secondary_encoder = hydra.utils.instantiate(
                self.hparams["model"].get("secondary_encoder", None),
            )
            if self.secondary_encoder is None:
                logger.warning("Secondary encoder could not be instantiated.")
        else:
            self.secondary_encoder = None

        if self.hparams["model"].get("PFN_for_ts", False) and len(tabular_attrs) > 1:
            self.ts_pfn_encoder = hydra.utils.instantiate(
                self.hparams["model"].get("ts_encoder", None),
            )
            if self.ts_pfn_encoder is None:
                logger.warning("TabPFN encoder for time-series could not be instantiated.")
        else:
            self.ts_pfn_encoder = None

        # Configure tokenizers and extract relevant info about the models' architectures
        self.nhead, self.separate_modality = check_model_encoder(self.tabular_encoder, self.hparams)

        # Initialize the time-series tokenizers if no TSPFN encoder is used
        self.time_series_tokenizer: TimeSeriesEmbedding | None
        if time_series_attrs:
            if isinstance(time_series_tokenizer, DictConfig):
                time_series_tokenizer_instance = hydra.utils.instantiate(time_series_tokenizer)
        else:
            # Set tokenizer to `None` if it's not going to be used
            time_series_tokenizer_instance = None
        self.time_series_tokenizer = time_series_tokenizer_instance

        self.time_series_processing_option = time_series_processing_option
        self.time_series_positional_encoding = time_series_positional_encoding
        self.fuse_ts_features = time_series_feature_fusion
        if self.fuse_ts_features:
            self.ts_feature_fusion = TimeSeriesFeatureFusion(n_time_series_attrs=self.n_time_series_attrs)

        self.mask_token = None

        # if perform_lora:
        #     adapter_encoder: nn.Module
        #     lora_linar = LoRALinear
        #     if not self.separate_modality:
        #         adapter_encoder = AdapterWrapperFT_Transformer(self.tabular_encoder, lora_linar, gamma=8, lora_alpha=8)
        #     elif isinstance(self.tabular_encoder, didactic.models.transformer.FT_Interleaved):
        #         adapter_encoder = AdapterWrapperFT_Interleaved(self.tabular_encoder, lora_linar, gamma=8, lora_alpha=8)
        #     else:
        #         adapter_encoder = AdapterWrapperFT_Transformer_CrossAtt(self.tabular_encoder, lora_linar, gamma=8, lora_alpha=8)
        #     setattr(self, "encoder", adapter_encoder)

        # Initialize inference storage tensors
        self.X_train_for_inference = torch.Tensor().to(self.device)
        self.ts_train_for_inference = torch.Tensor().to(self.device)
        self.y_train_for_inference = torch.Tensor().to(self.device)

    def on_validation_epoch_start(self):
        # Assumes the first dataloader is the train loader
        train_loader = self.trainer.train_dataloader
        if train_loader is None:
            print("Validation sanity check")
        else:
            if callable(train_loader):
                train_loader = train_loader()
            tabular_attrs_for_train_inference = {attr: [] for attr in self.hparams["tabular_attrs"]}
            for batch in train_loader:
                for attr in self.hparams["tabular_attrs"]:
                    tabular_attrs_for_train_inference[attr].append(batch[attr])
            tabular_attrs_for_train_inference = {
                attr: torch.cat(tabular_attrs_for_train_inference[attr], dim=0)
                for attr in tabular_attrs_for_train_inference
            }

            y_train_for_inference = tabular_attrs_for_train_inference.pop(self.target_attr)
            if self.time_series_processing_option == "collapse":
                y_train_for_inference = y_train_for_inference.repeat_interleave(repeats=self.n_time_series_attrs, dim=0)
            self.y_train_for_inference = y_train_for_inference.to(self.device)

            if len(tabular_attrs_for_train_inference) > 0:
                X_train_for_inference = torch.hstack(
                    [tabular_attrs_for_train_inference[attr].unsqueeze(1) for attr in tabular_attrs_for_train_inference]
                )
                X_train_for_inference = X_train_for_inference.unsqueeze(1)
                self.X_train_for_inference = X_train_for_inference.to(self.device)

            if self.hparams["model"].get("PFN_for_ts", False) and self.hparams["time_series_attrs"]:
                ts_attrs_for_train_inference = {}
                for batch in train_loader:
                    ts_for_train_inference = {
                        attr: attr_data
                        for attr, attr_data in filter_time_series_attributes(
                            batch, views=self.hparams["views"], attrs=self.hparams["time_series_attrs"]
                        )[0].items()
                    }
                    for attr, view in ts_for_train_inference.keys():
                        ts_attrs_for_train_inference.setdefault((attr, view), []).append(
                            ts_for_train_inference[(attr, view)]
                        )

                ts_cat_attrs_for_train_inference = {
                    attr: torch.cat(ts_attrs_for_train_inference[attr], dim=0) for attr in ts_attrs_for_train_inference
                }
                ts_data: Dict[Tuple[str, str], List[Tensor]] = {}
                for cross_attrs in ts_cat_attrs_for_train_inference:
                    cross_attrs_column: Tuple[str, str] = (cross_attrs[0].__str__(), cross_attrs[1].__str__())
                    ts_data_value = F.interpolate(
                        ts_cat_attrs_for_train_inference[cross_attrs].unsqueeze(1), size=64, mode="linear"
                    ).squeeze(1)
                    ts_data.setdefault(cross_attrs_column, []).append(ts_data_value)
                ts_data_stacked = {
                    ts_view_attr: torch.vstack(batch_vals) for ts_view_attr, batch_vals in ts_data.items()
                }
                ts_data_doublestack = {
                    ts_view: torch.hstack(
                        [ts_data_stacked[(view, attr)] for (view, attr) in ts_data_stacked.keys() if view == ts_view]
                    )
                    for ts_view in self.hparams["views"]
                }
                ts_tokens_validation_list = []
                for ts_view in ts_data_doublestack:
                    ts_view_data = ts_data_doublestack[ts_view]
                    ts_tokens_validation_list.append(ts_view_data)
                ts_tokens_validation = torch.stack(
                    ts_tokens_validation_list, dim=1
                ).float()  # (N, 1, S_ts * num_time_units)
                if self.time_series_processing_option == "collapse":
                    ts_tokens_validation = ts_collapse_on_features(
                        ts_tokens_validation,
                        n_time_series_attrs=self.n_time_series_attrs,
                    )
                if self.fuse_ts_features:
                    # Use TS fusion module
                    ts_tokens_validation = self.ts_feature_fusion(ts_tokens_validation)
                self.ts_train_for_inference = ts_tokens_validation.to(self.device)

    def on_test_epoch_start(self):
        train_loader = self.trainer.test_dataloaders[1]
        # val_loader = self.trainer.test_dataloaders[2]
        if train_loader is None:
            raise ValueError("No training dataloader found while setting up inference storage tensors.")
        else:
            if callable(train_loader):
                train_loader = train_loader()
            tabular_attrs_for_train_inference = {attr: [] for attr in self.hparams["tabular_attrs"]}
            for batch in train_loader:
                for attr in self.hparams["tabular_attrs"]:
                    tabular_attrs_for_train_inference[attr].append(batch[attr])
            tabular_attrs_for_train_inference = {
                attr: torch.cat(tabular_attrs_for_train_inference[attr], dim=0)
                for attr in tabular_attrs_for_train_inference
            }

            y_train_for_inference = tabular_attrs_for_train_inference.pop(self.target_attr)
            if self.time_series_processing_option == "collapse":
                y_train_for_inference = y_train_for_inference.repeat_interleave(repeats=self.n_time_series_attrs, dim=0)
            self.y_train_for_inference = y_train_for_inference.to(self.device)

            if len(tabular_attrs_for_train_inference) > 0:
                X_train_for_inference = torch.hstack(
                    [tabular_attrs_for_train_inference[attr].unsqueeze(1) for attr in tabular_attrs_for_train_inference]
                )
                X_train_for_inference = X_train_for_inference.unsqueeze(1)
                self.X_train_for_inference = X_train_for_inference.to(self.device)

            if self.hparams["model"].get("PFN_for_ts", False) and self.hparams["time_series_attrs"]:
                ts_attrs_for_train_inference = {}
                for batch in train_loader:
                    ts_for_train_inference = {
                        attr: attr_data
                        for attr, attr_data in filter_time_series_attributes(
                            batch, views=self.hparams["views"], attrs=self.hparams["time_series_attrs"]
                        )[0].items()
                    }
                    for attr, view in ts_for_train_inference.keys():
                        ts_attrs_for_train_inference.setdefault((attr, view), []).append(
                            ts_for_train_inference[(attr, view)]
                        )

                ts_cat_attrs_for_train_inference = {
                    attr: torch.cat(ts_attrs_for_train_inference[attr], dim=0) for attr in ts_attrs_for_train_inference
                }
                ts_data: Dict[Tuple[str, str], List[Tensor]] = {}
                for cross_attrs in ts_cat_attrs_for_train_inference:
                    cross_attrs_column: Tuple[str, str] = (cross_attrs[0].__str__(), cross_attrs[1].__str__())
                    ts_data_value = F.interpolate(
                        ts_cat_attrs_for_train_inference[cross_attrs].unsqueeze(1), size=64, mode="linear"
                    ).squeeze(1)
                    ts_data.setdefault(cross_attrs_column, []).append(ts_data_value)
                ts_data_stacked = {
                    ts_view_attr: torch.vstack(batch_vals) for ts_view_attr, batch_vals in ts_data.items()
                }
                ts_data_doublestack = {
                    ts_view: torch.hstack(
                        [ts_data_stacked[(view, attr)] for (view, attr) in ts_data_stacked.keys() if view == ts_view]
                    )
                    for ts_view in self.hparams["views"]
                }
                ts_tokens_test_list = []
                for ts_view in ts_data_doublestack:
                    ts_view_data = ts_data_doublestack[ts_view]
                    ts_tokens_test_list.append(ts_view_data)
                ts_tokens_test = torch.stack(ts_tokens_test_list, dim=1).float()  # (N, 1, S_ts * num_time_units)
                if self.time_series_processing_option == "collapse":
                    ts_tokens_test = ts_collapse_on_features(
                        ts_tokens_test,
                        n_time_series_attrs=self.n_time_series_attrs,
                    )
                if self.fuse_ts_features:
                    # Use TS fusion module
                    ts_tokens_test = self.ts_feature_fusion(ts_tokens_test)
                self.ts_train_for_inference = ts_tokens_test.to(self.device)

        # if val_loader is None:
        #     raise ValueError("No validation dataloader found while setting up inference storage tensors.")
        # else:
        #     if callable(val_loader):
        #         val_loader = val_loader()
        #     tabular_attrs_for_val_inference = {attr: [] for attr in self.hparams["tabular_attrs"]}
        #     for batch in val_loader:
        #         for attr in self.hparams["tabular_attrs"]:
        #             tabular_attrs_for_val_inference[attr].append(batch[attr])
        #     tabular_attrs_for_val_inference = {
        #         attr: torch.cat(tabular_attrs_for_val_inference[attr], dim=0)
        #         for attr in tabular_attrs_for_val_inference
        #     }

        #     y_val_for_inference = tabular_attrs_for_val_inference.pop(self.target_attr)
        #     self.y_train_for_inference = torch.cat(
        #         [self.y_train_for_inference, y_val_for_inference.to(self.device)],
        #         dim=0
        #     )

        #     if len(tabular_attrs_for_val_inference) > 0:
        #         X_val_for_inference = torch.hstack(
        #             [tabular_attrs_for_val_inference[attr].unsqueeze(1) for attr in tabular_attrs_for_val_inference]
        #         )
        #         X_val_for_inference = X_val_for_inference.unsqueeze(1)
        #         self.X_train_for_inference = torch.cat(
        #             [self.X_train_for_inference, X_val_for_inference.to(self.device)],
        #             dim=0
        #         )

        #     if self.hparams["model"].get("PFN_for_ts", False) and self.hparams["time_series_attrs"]:
        #         ts_attrs_for_val_inference = {}
        #         for batch in val_loader:
        #             ts_for_val_inference = {
        #                 attr: attr_data
        #                 for attr, attr_data in filter_time_series_attributes(
        #                     batch, views=self.hparams["views"], attrs=self.hparams["time_series_attrs"]
        #                 )[0].items()
        #             }
        #             for attr, view in ts_for_val_inference.keys():
        #                 ts_attrs_for_val_inference.setdefault((attr, view), []).append(
        #                     ts_for_val_inference[(attr, view)]
        #                 )

        #         ts_cat_attrs_for_val_inference = {
        #             attr: torch.cat(ts_attrs_for_val_inference[attr], dim=0) for attr in ts_attrs_for_val_inference
        #         }
        #         ts_data: Dict[Tuple[str, str], List[Tensor]] = {}
        #         for cross_attrs in ts_cat_attrs_for_val_inference:
        #             cross_attrs_column: Tuple[str, str] = (cross_attrs[0].__str__(), cross_attrs[1].__str__())
        #             ts_data_value = F.interpolate(
        #                 ts_cat_attrs_for_val_inference[cross_attrs].unsqueeze(1), size=64, mode="linear"
        #             ).squeeze(1)
        #             ts_data.setdefault(cross_attrs_column, []).append(ts_data_value)
        #         ts_data_stacked = {
        #             ts_view_attr: torch.vstack(batch_vals) for ts_view_attr, batch_vals in ts_data.items()
        #         }
        #         ts_data_doublestack = {
        #             ts_view: torch.hstack(
        #                 [ts_data_stacked[(view, attr)] for (view, attr) in ts_data_stacked.keys() if view == ts_view]
        #             )
        #             for ts_view in self.hparams["views"]
        #         }
        #         ts_tokens_list = []
        #         for ts_view in ts_data_doublestack:
        #             ts_view_data = ts_data_doublestack[ts_view]
        #             ts_tokens_list.append(ts_view_data)
        #         ts_tokens = torch.stack(ts_tokens_list, dim=1).float()  # (N, 1, S_ts * num_time_units)
        #         self.ts_train_for_inference = torch.cat(
        #             [self.ts_train_for_inference, ts_tokens.to(self.device)],
        #             dim=0
        #         )

    def on_predict_epoch_start(self):
        train_loader = self.trainer.predict_dataloaders[1]
        if train_loader is None:
            raise ValueError("No training dataloader found while setting up inference storage tensors.")
        else:
            if callable(train_loader):
                train_loader = train_loader()
            tabular_attrs_for_train_inference = {attr: [] for attr in self.hparams["tabular_attrs"]}
            for batch in train_loader:
                for attr in self.hparams["tabular_attrs"]:
                    tabular_attrs_for_train_inference[attr].append(batch[attr])
            tabular_attrs_for_train_inference = {
                attr: torch.cat(tabular_attrs_for_train_inference[attr], dim=0)
                for attr in tabular_attrs_for_train_inference
            }

            y_train_for_inference = tabular_attrs_for_train_inference.pop(self.target_attr)
            if self.time_series_processing_option == "collapse":
                y_train_for_inference = y_train_for_inference.repeat_interleave(repeats=self.n_time_series_attrs, dim=0)
            self.y_train_for_inference = y_train_for_inference.to(self.device)

            if len(tabular_attrs_for_train_inference) > 0:
                X_train_for_inference = torch.hstack(
                    [tabular_attrs_for_train_inference[attr].unsqueeze(1) for attr in tabular_attrs_for_train_inference]
                )
                X_train_for_inference = X_train_for_inference.unsqueeze(1)
                self.X_train_for_inference = X_train_for_inference.to(self.device)

            if self.hparams["model"].get("PFN_for_ts", False) and self.hparams["time_series_attrs"]:
                ts_attrs_for_train_inference = {}
                for batch in train_loader:
                    ts_for_train_inference = {
                        attr: attr_data
                        for attr, attr_data in filter_time_series_attributes(
                            batch, views=self.hparams["views"], attrs=self.hparams["time_series_attrs"]
                        )[0].items()
                    }
                    for attr, view in ts_for_train_inference.keys():
                        ts_attrs_for_train_inference.setdefault((attr, view), []).append(
                            ts_for_train_inference[(attr, view)]
                        )

                ts_cat_attrs_for_train_inference = {
                    attr: torch.cat(ts_attrs_for_train_inference[attr], dim=0) for attr in ts_attrs_for_train_inference
                }
                ts_data: Dict[Tuple[str, str], List[Tensor]] = {}
                for cross_attrs in ts_cat_attrs_for_train_inference:
                    cross_attrs_column: Tuple[str, str] = (cross_attrs[0].__str__(), cross_attrs[1].__str__())
                    ts_data_value = F.interpolate(
                        ts_cat_attrs_for_train_inference[cross_attrs].unsqueeze(1), size=64, mode="linear"
                    ).squeeze(1)
                    ts_data.setdefault(cross_attrs_column, []).append(ts_data_value)
                ts_data_stacked = {
                    ts_view_attr: torch.vstack(batch_vals) for ts_view_attr, batch_vals in ts_data.items()
                }
                ts_data_doublestack = {
                    ts_view: torch.hstack(
                        [ts_data_stacked[(view, attr)] for (view, attr) in ts_data_stacked.keys() if view == ts_view]
                    )
                    for ts_view in self.hparams["views"]
                }
                ts_tokens_prediction_list = []
                for ts_view in ts_data_doublestack:
                    ts_view_data = ts_data_doublestack[ts_view]
                    ts_tokens_prediction_list.append(ts_view_data)
                ts_tokens_prediction = torch.stack(
                    ts_tokens_prediction_list, dim=1
                ).float()  # (N, 1, S_ts * num_time_units)
                if self.time_series_processing_option == "collapse":
                    ts_tokens_prediction = ts_collapse_on_features(
                        ts_tokens_prediction,
                        n_time_series_attrs=self.n_time_series_attrs,
                    )
                if self.fuse_ts_features:
                    # Use TS fusion module
                    ts_tokens_prediction = self.ts_feature_fusion(ts_tokens_prediction)
                self.ts_train_for_inference = ts_tokens_prediction.to(self.device)

    @property
    def example_input_array(
        self,
    ) -> Tuple[Dict[TabularAttribute, Tensor], Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor]]:
        """Redefine example input array based on the cardiac attributes provided to the model."""
        # 2 is the size of the batch in the example
        tab_attrs = {}
        # Only generate 0/1 labels, to avoid generating labels bigger than the number of classes, which would lead to
        # an index out of range error when looking up the embedding of the class in the categorical feature tokenizer
        for attr in self.hparams["tabular_attrs"]:
            if attr != self.target_attr:
                if attr in self.tabular_cat_attrs:
                    cardinality = len(TABULAR_CAT_ATTR_LABELS[attr]) - 1
                    tab_attrs[attr] = torch.randint(cardinality, (10,))
                else:
                    tab_attrs[attr] = torch.randn(10)
        labels = torch.randperm(5)
        tab_attrs.update({self.target_attr: torch.cat([labels, labels])})
        time_series_attrs = {
            (view, attr): torch.randn(10, self.hparams["data_params"].in_shape[OrchidTag.time_series_attrs][1])
            for view, attr in itertools.product(self.hparams["views"], self.hparams["time_series_attrs"])
        }
        time_series_notna_mask = torch.ones(
            (10, len(self.hparams["views"]) * len(self.hparams["time_series_attrs"])), dtype=torch.bool
        )
        return tab_attrs, time_series_attrs, time_series_notna_mask

    def configure_model(
        self,
    ) -> Tuple[nn.Module, nn.Module, Optional[nn.ModuleDict]]:
        """Build the model, which must return a transformer encoder, and self-supervised or prediction heads."""
        # Build the transformer encoder
        tabular_encoder = hydra.utils.instantiate(self.hparams["model"]["encoder"])

        # Build the projection head for contrastive learning, if contrastive learning is enabled
        # contrastive_head = None
        # if self.contrastive_loss and not self.orthogonal_loss:
        #     contrastive_head = hydra.utils.instantiate(self.hparams["model"]["contrastive_head"])
        # elif self.contrastive_loss and self.orthogonal_loss:
        print("Temporary set contrastive head to identity")
        contrastive_head = nn.Identity()

        # Build the prediction heads (one by tabular attribute to predict) following the architecture proposed in
        # https://arxiv.org/pdf/2106.11959
        prediction_heads = None
        if self.predict_losses:
            prediction_heads = nn.ModuleDict()
            for target_tab_attr in self.predict_losses:
                if (
                    target_tab_attr in TabularAttribute.categorical_attrs()
                    and target_tab_attr not in TabularAttribute.binary_attrs()
                ):
                    # Multi-class classification target
                    output_size = len(TABULAR_CAT_ATTR_LABELS[target_tab_attr])
                else:
                    # Binary classification or regression target
                    output_size = 1

                prediction_heads[target_tab_attr] = hydra.utils.instantiate(
                    self.hparams["model"]["prediction_head"], out_features=output_size
                )

        return tabular_encoder, contrastive_head, prediction_heads

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizer to ignore parameters that should remain frozen (e.g. tokenizers)."""
        # Frozen tabpfn encoder
        if self.hparams["model"].get("freeze_encoder", False):
            for param in self.tabular_encoder.parameters():
                param.requires_grad = False
        if self.hparams["model"].get("freeze_ts_pfn_encoder", False) and self.ts_pfn_encoder is not None:
            for param in self.ts_pfn_encoder.parameters():
                param.requires_grad = False
        return super().configure_optimizers()

    @auto_move_data
    def process_data(
        self,
        tabular_attrs: Dict[TabularAttribute, Tensor],
        time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor],
        time_series_notna_mask: Optional[Tensor] = None,
        leave_one_out_index: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Tokenizes the input tabular and time-series attributes, providing a mask of non-missing attributes.

        Args:
            tabular_attrs: (K: S, V: N), Sequence of batches of tabular attributes. To indicate an item is missing an
                attribute, the flags `MISSING_NUM_ATTR`/`MISSING_CAT_ATTR` can be used for numerical and categorical
                attributes, respectively.
            time_series_attrs: (K: S, V: (N, ?)), Sequence of batches of time-series attributes, where the
                dimensionality of each attribute can vary.

        Returns:
            Batch of i) (N, S, E) tokens for each attribute, and ii) (N, S) mask of non-missing attributes.
        """
        # Initialize lists for cumulating (optional) tensors for each modality, that will be concatenated into tensors
        ts_tokens_list, notna_mask_list = [], []
        assert len(tabular_attrs) > 1 or self.hparams["model"].get(
            "PFN_for_ts", False
        ), "At least one tabular attribute is required if PFN_for_ts is not enabled."
        # Tokenize the attributes
        if time_series_attrs:
            if self.hparams["model"].get("PFN_for_ts", False):
                ts_data: Dict[Tuple[str, str], List[Tensor]] = {}
                for cross_attrs in time_series_attrs:
                    cross_attrs_column: Tuple[str, str] = (cross_attrs[0].__str__(), cross_attrs[1].__str__())
                    ts_data_value = F.interpolate(
                        time_series_attrs[cross_attrs].unsqueeze(1), size=64, mode="linear"
                    ).squeeze(1)
                    ts_data.setdefault(cross_attrs_column, []).append(ts_data_value)
                ts_data_stacked = {
                    ts_view_attr: torch.vstack(batch_vals) for ts_view_attr, batch_vals in ts_data.items()
                }
                ts_data_doublestack = {
                    ts_view: torch.hstack(
                        [ts_data_stacked[(view, attr)] for (view, attr) in ts_data_stacked.keys() if view == ts_view]
                    )
                    for ts_view in self.hparams["views"]
                }
                for ts_view in ts_data_doublestack:
                    ts_view_data = ts_data_doublestack[ts_view]
                    ts_tokens_list.append(ts_view_data)

                ts_tokens = torch.stack(ts_tokens_list, dim=1)  # (N, 1, S_ts * num_time_units)

            else:
                assert (
                    self.time_series_tokenizer is not None
                ), "Time-series tokenizer must be configured to tokenize time-series."
                time_series_attrs_tokens = self.time_series_tokenizer(time_series_attrs)  # S * (N, ?) -> (N, S_ts, E)
                ts_tokens_list.extend(time_series_attrs_tokens)
                ts_tokens = torch.stack(ts_tokens_list)  # (N, S_ts, E)

            # Cast to float to make sure tokens are not represented using double
            ts_tokens = ts_tokens.float()

            # Indicate that, when time-series tokens are requested, they are always available
            if time_series_notna_mask is None:
                time_series_notna_mask = torch.full(ts_tokens.shape[:2], True, device=ts_tokens.device)

            # if self.hparams["model"].get("PFN_for_ts", False):
            indices = list(range(len(ts_tokens)))
            y = tabular_attrs[self.target_attr]
            if self.training or len(self.y_train_for_inference) == 0:
                if self.hparams["split_finetuning"] > 0.0:
                    label = y.clone().cpu()
                    try:
                        train_indices, test_indices = train_test_split(
                            indices,
                            test_size=self.hparams["split_finetuning"],
                            random_state=self.hparams["seed"],
                            stratify=label,
                        )
                    except ValueError:
                        if len(indices) == 1:
                            print("Sanity check with single sample for TS, skipping train/test split.")
                            train_indices, test_indices = indices, indices
                        else:
                            print("Stratified splitting failed, performing non-stratified split instead.")
                            train_indices, test_indices = train_test_split(
                                indices,
                                test_size=self.hparams["split_finetuning"],
                                random_state=self.hparams["seed"],
                            )

                else:
                    assert (
                        leave_one_out_index is not None
                    ), "When split_finetuning is 0.0, leave_one_out_index must be set."
                    # Leave one out
                    train_indices = indices[:leave_one_out_index] + indices[leave_one_out_index + 1 :]
                    test_indices = indices[leave_one_out_index : leave_one_out_index + 1]

                ts_tokens_support = torch.as_tensor(ts_tokens[train_indices], dtype=torch.float32)
                ts_tokens_query = torch.as_tensor(ts_tokens[test_indices], dtype=torch.float32)
                ts_tokens_support = ts_tokens_support.unsqueeze(0) if ts_tokens_support.ndim == 2 else ts_tokens_support
                ts_tokens_query = ts_tokens_query.unsqueeze(0) if ts_tokens_query.ndim == 2 else ts_tokens_query

                # ts_tokens_full = torch.cat([ts_tokens_support, ts_tokens_query], dim=0)
                ts_tokens = torch.cat([ts_tokens_support, ts_tokens_query], dim=0)
                # ts_tokens = ts_tokens_full.unsqueeze(1)  # (N, 1, S_ts)
            # else:
            #     ts_tokens = ts_tokens.unsqueeze(1)  # (N, 1, S_ts)

            notna_mask_list.extend(time_series_notna_mask)

            # Cast to bool to make sure attention mask is represented by bool
            notna_mask = torch.stack(notna_mask_list).squeeze(1).bool()  # (N, S_ts)

        X_batch_full = None
        if tabular_attrs and len(tabular_attrs) > 1:  # More than just the target

            y = tabular_attrs.pop(self.target_attr)
            label = y.clone().cpu()

            X = torch.hstack([tabular_attrs[attr].unsqueeze(1) for attr in tabular_attrs])

            # Preprocess tabular data (impute missing values, one-hot encode categorical features, standardize numerical
            # features, etc.)
            # X_preprocessed = preprocess_tensor(X, self.cat_idxs, self.tabular_cat_attrs_cardinalities)
            X_preprocessed = torch.as_tensor(X, dtype=torch.float32)  # TOFIX: Disable preprocessing for TabPFN
            # X_preprocessed = X_input_preprocessing(X, self.cat_idxs).to(self.device)
            # X = torch.as_tensor(X_preprocessed, dtype=torch.float32)

            if self.training or len(self.X_train_for_inference) == 0:

                indices = list(range(len(y)))
                if self.hparams["split_finetuning"] > 0.0:
                    try:
                        train_indices, test_indices = train_test_split(
                            indices,
                            test_size=self.hparams["split_finetuning"],
                            random_state=self.hparams["seed"],
                            stratify=label,
                        )
                    except ValueError:
                        if len(indices) == 1:
                            print("Sanity check with single sample, skipping train/test split.")
                            train_indices, test_indices = indices, indices
                        else:
                            print("Stratified splitting failed, performing non-stratified split instead.")
                            train_indices, test_indices = train_test_split(
                                indices,
                                test_size=self.hparams["split_finetuning"],
                                random_state=self.hparams["seed"],
                            )
                else:
                    assert (
                        leave_one_out_index is not None
                    ), "When split_finetuning is 0.0, leave_one_out_index must be set."
                    # Leave one out
                    train_indices = indices[:leave_one_out_index] + indices[leave_one_out_index + 1 :]
                    test_indices = indices[leave_one_out_index : leave_one_out_index + 1]

                X_batch_support = torch.as_tensor(X_preprocessed[train_indices], dtype=torch.float32)
                X_batch_query = torch.as_tensor(X_preprocessed[test_indices], dtype=torch.float32)

                X_batch_query = X_batch_query.unsqueeze(0) if X_batch_query.ndim == 1 else X_batch_query

                X_batch_full = torch.cat([X_batch_support, X_batch_query], dim=0).unsqueeze(1)

            else:
                X_batch_full = X_preprocessed.unsqueeze(1)

        if self.training or len(self.y_train_for_inference) == 0:
            y_batch_support = torch.as_tensor(y[train_indices], dtype=torch.float32)
            y_batch_query = torch.as_tensor(y[test_indices], dtype=torch.float32)
        else:
            y_batch_support = y.to(self.device)
            y_batch_query = y.to(self.device)

        if not time_series_attrs:
            ts_tokens = torch.empty(
                (X_batch_full.shape[0], 0, self.hparams["embed_dim"]),
                device=X_batch_full.device,
            )
            notna_mask = torch.empty((X_batch_full.shape[0], 0), device=X_batch_full.device, dtype=torch.bool)

        if ts_tokens.ndim == 2:
            ts_tokens = ts_tokens.unsqueeze(0)

        if self.time_series_processing_option == "collapse":
            ts_tokens = ts_collapse_on_features(
                ts_tokens,
                n_time_series_attrs=self.n_time_series_attrs,
            )
            # Repeat other tensors accordingly
            if X_batch_full is not None:
                X_batch_full = X_batch_full.repeat_interleave(repeats=self.n_time_series_attrs, dim=0)
            y_batch_support = y_batch_support.repeat_interleave(repeats=self.n_time_series_attrs, dim=0)
            # y_batch_query = y_batch_query.repeat_interleave(repeats=self.n_time_series_attrs, dim=0)
            # notna_mask = notna_mask.repeat_interleave(repeats=self.n_time_series_attrs, dim=0)

        return (
            X_batch_full,
            y_batch_support,
            y_batch_query,
            ts_tokens,
            notna_mask,
        )

    def preprocess_tokens(self, tokens: Tensor, avail_mask: Tensor, enable_augments: bool = False) -> Tensor:
        """Preprocesses the input tokens, optionally masking missing data and random tokens to cause perturbations.

        Args:
            tokens: (N, S, E) Tokens to preprocess.
            avail_mask: (N, S), Boolean mask indicating available (i.e. non-missing) tokens.
            enable_augments: Whether to perform augments on the tokens (e.g. masking) to obtain a "corrupted" view for
                contrastive learning. Augments are already configured differently for training/testing (to avoid
                stochastic test-time predictions), so this parameter is simply useful to easily toggle augments on/off
                to obtain contrasting views.

        Returns:
            Tokens with missing data masked and/or random tokens replaced by the mask token.
        """
        # mask_token = self.mask_token
        mask_token: Tensor

        if self.mask_token is None:
            return tokens

        if isinstance(self.mask_token, ParameterDict):
            mask_token = torch.stack(list(self.mask_token.values()))
        else:
            assert isinstance(self.mask_token, Parameter), "Mask token must be a Parameter or ParameterDict."
            mask_token = cast(Tensor, self.mask_token)

        # If a mask token is configured, substitute the missing tokens with the mask token to distinguish them from
        # the other tokens
        tokens = mask_tokens(tokens, mask_token, ~avail_mask)

        mtr_p = self.train_mtr_p if self.training else self.test_mtr_p
        if mtr_p and enable_augments:
            # Mask Token Replacement (MTR) data augmentation
            # Replace random non-missing tokens with the mask token to perturb the input
            tokens, _ = random_masking(tokens, mask_token, mtr_p)

        return tokens

    @auto_move_data
    def encode(
        self,
        X_batch_full: Tensor,
        y_batch_support: Tensor,
        ts_tokens: Tensor,
        avail_mask: Tensor,
        enable_augments: bool = False,
    ) -> Tensor:
        """Embeds input sequences using the encoder model, optionally selecting/pooling output ts_tokens for the embedding.

        Args:
            ts_tokens: (N, S, E), Tokens to feed to the encoder.
            avail_mask: (N, S), Boolean mask indicating available (i.e. non-missing) ts_tokens. Missing ts_tokens can thus be
                treated distinctly from others (e.g. replaced w/ a specific mask).
            enable_augments: Whether to perform augments on the ts_tokens (e.g. masking) to obtain a "corrupted" view for
                contrastive learning. Augments are already configured differently for training/testing (to avoid
                stochastic test-time predictions), so this parameter is simply useful to easily toggle augments on/off
                to obtain contrasting views.

        Returns: (N, E), Embeddings of the input sequences.
        """

        if not self.hparams["model"].get("PFN_for_ts", False):
            ts_tokens = self.preprocess_tokens(ts_tokens, avail_mask, enable_augments=enable_augments)
        else:
            if self.fuse_ts_features:
                # Use TS fusion module
                ts_tokens = self.ts_feature_fusion(ts_tokens)

        if X_batch_full is None:
            assert self.hparams["model"].get("PFN_for_ts", False), "X_batch_full is None but PFN_for_ts is not enabled."
            if self.training or len(self.ts_train_for_inference) == 0:
                out_features = self.tabular_encoder(
                    ts_tokens, y_batch_support, use_ts_sinus=self.time_series_positional_encoding
                )[:, -1, :]
            else:
                # Use train set as context for predicting the query set
                ts_tokens = torch.cat([self.ts_train_for_inference, ts_tokens], dim=0)
                y_train = self.y_train_for_inference
                out_features = self.tabular_encoder(
                    ts_tokens, y_train, use_ts_sinus=self.time_series_positional_encoding
                )[:, -1, :]

            # if self.secondary_encoder is not None:
            #     out_features = self.secondary_encoder(out_features)
            # return out_features  # (N, d_model)

        elif X_batch_full is not None:
            if self.training or len(self.X_train_for_inference) == 0:
                out_tabular_features = self.tabular_encoder(X_batch_full, y_batch_support)
                if self.hparams["model"].get("PFN_for_ts", False):
                    assert self.ts_pfn_encoder is not None
                    out_ts_features = self.ts_pfn_encoder(
                        ts_tokens, y_batch_support, use_ts_sinus=self.time_series_positional_encoding
                    )
                else:
                    out_ts_features = ts_tokens.squeeze(1)  # (N, S, E)
                    out_ts_features = out_ts_features[len(y_batch_support) :]  # (N_query, S, E)
            else:
                # Use train set as context for predicting the query set
                X_full_train_test = torch.cat([self.X_train_for_inference, X_batch_full], dim=0)
                y_train = self.y_train_for_inference
                out_tabular_features = self.tabular_encoder(X_full_train_test, y_train)
                if self.hparams["model"].get("PFN_for_ts", False):
                    assert self.ts_pfn_encoder is not None
                    ts_tokens = torch.cat([self.ts_train_for_inference, ts_tokens], dim=0)
                    out_ts_features = self.ts_pfn_encoder(
                        ts_tokens, y_train, use_ts_sinus=self.time_series_positional_encoding
                    )
                else:
                    out_ts_features = ts_tokens.squeeze(1)  # (N, S, E)

            if self.fusion_module is None:
                return out_tabular_features[:, -1]

            elif isinstance(self.fusion_module, didactic.models.fusionners.MLPFusion):
                out_tabular_features = out_tabular_features[:, -1]  # (N, E)
                out_tabular_features = (
                    self.secondary_encoder(out_tabular_features)
                    if self.secondary_encoder is not None
                    else out_tabular_features
                )
                if self.hparams["model"].get("PFN_for_ts", False):
                    out_ts_features = out_ts_features[:, -1, :]  # (N, E)
                else:
                    out_ts_features = out_ts_features.mean(dim=1)  # (N, E)
                intermediate_features = torch.cat([out_tabular_features, out_ts_features], dim=1)  # (N, S, 2E)
                out_features = self.fusion_module(intermediate_features)

            elif isinstance(self.fusion_module, didactic.models.transformer.FT_Interleaved_Alignment):
                if self.hparams["model"].get("PFN_for_ts", False):
                    # Do not use twice (tabular and ts) the label representation token
                    out_ts_features = out_ts_features[:, :-1, :]
                out_features = self.fusion_module(out_tabular_features, out_ts_features)[
                    :, -1, :
                ]  # (N, S, E) -> (N, E)

            else:
                raise ValueError(f"Unknown fusion module '{self.fusion_module}'.")

        return out_features  # (N, d_model)

    @auto_move_data
    def forward(
        self,
        tabular_attrs: Dict[TabularAttribute, Tensor],
        time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor],
        time_series_notna_mask: Optional[Tensor],
        task: Literal["encode", "predict", "continuum_param", "continuum_tau"] = "encode",
    ) -> Tensor | Dict[TabularAttribute, Tensor]:
        """Performs a forward pass through i) the tokenizer, ii) the transformer encoder and iii) the prediction head.

        Args:
            tabular_attrs: (K: S, V: N) Sequence of batches of tabular attributes. To indicate an item is missing an
                attribute, the flags `MISSING_NUM_ATTR`/`MISSING_CAT_ATTR` can be used for numerical and categorical
                attributes, respectively.
            time_series_attrs: (K: S, V: (N, ?)), Sequence of batches of time-series attributes, where the
                dimensionality of each attribute can vary.
            task: Flag indicating which type of inference task to perform.

        Returns:
            if `task` == 'encode':
                (N, E), Batch of features extracted by the encoder.
            if `task` == 'continuum_param`:
                ? * (M), Parameter of the unimodal logits distribution for targets.
            if `task` == 'continuum_tau`:
                ? * (M), Temperature used to control the sharpness of the unimodal logits distribution for targets.
            if `task` == 'predict' (and the model includes prediction heads):
                ? * (N), Prediction for each target in `losses`.
        """
        if task != "encode" and not self.prediction_heads:
            raise ValueError(
                "You requested to perform a prediction task, but the model does not include any prediction heads."
            )
        tabular_attrs = {
            attr: attr_data.unsqueeze(0) if attr_data.ndim == 0 else attr_data
            for attr, attr_data in tabular_attrs.items()
            if attr in self.hparams["tabular_attrs"]
        }
        time_series_attrs = {
            attr: attr_data.unsqueeze(0) if attr_data.ndim == 1 else attr_data
            for attr, attr_data in time_series_attrs.items()
        }
        if time_series_notna_mask is not None:
            time_series_notna_mask = (
                time_series_notna_mask.unsqueeze(0) if time_series_notna_mask.ndim == 1 else time_series_notna_mask
            )
        # TOFIX: leave_one_out_index should be passed as argument
        X_batch_full, y_batch_support, y_batch_query, ts_tokens, avail_mask = self.process_data(
            tabular_attrs,
            time_series_attrs,
            time_series_notna_mask,
            leave_one_out_index=0 if self.hparams["split_finetuning"] == 0.0 else None,
        )  # (N, S, E), (N, S)

        out_features = self.encode(X_batch_full, y_batch_support, ts_tokens, avail_mask)  # (N, S, E) -> (N, E)

        # Early return if requested task requires no prediction heads
        if task == "encode":
            if self.time_series_processing_option == "collapse":
                # Average out the latent vectors across time-series attributes
                n_attrs = self.n_time_series_attrs
                out_features = out_features.view(-1, n_attrs, out_features.shape[-1]).mean(dim=1)
            return out_features

        assert (
            self.prediction_heads is not None
        ), "You requested to perform a prediction task, but the model does not include any prediction heads."
        # Forward pass through each target's prediction head
        predictions = {attr: prediction_head(out_features) for attr, prediction_head in self.prediction_heads.items()}

        # Based on the requested task, extract and format the appropriate output of the prediction heads
        match task:
            case "predict":
                pass
            case "continuum_param":
                predictions = {attr: pred[1] for attr, pred in predictions.items()}
            case "continuum_tau":
                predictions = {attr: pred[2] for attr, pred in predictions.items()}
            case _:
                raise ValueError(f"Unknown task '{task}'.")

        # Squeeze out the singleton dimension from the predictions' features (only relevant for scalar predictions)
        predictions = {attr: prediction.squeeze(dim=1) for attr, prediction in predictions.items()}
        if self.time_series_processing_option == "collapse":
            # Average out the predictions across time-series attributes
            n_attrs = self.n_time_series_attrs
            predictions = {
                attr: prediction.view(-1, n_attrs, prediction.shape[-1]).mean(dim=1)
                for attr, prediction in predictions.items()
            }
        return predictions

    @auto_move_data
    def get_latent_vectors(
        self,
        batch: PatientData,
        batch_idx: int,
        tabular_attrs: Dict[TabularAttribute, Tensor],
        time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor],
    ) -> Tensor:
        """Extracts the latent vectors from the encoder for the given batch."""
        X_batch_full, y_batch_support, y_batch_query, ts_tokens, avail_mask = self.process_data(
            tabular_attrs, time_series_attrs, None
        )  # (N, S, E), (N, S)
        return self.encode(
            X_batch_full,
            y_batch_support,
            y_batch_query,
            ts_tokens,
            avail_mask,
        )

    def _shared_step(self, batch: PatientDataTarget, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Tensor]:
        # Extract tabular and time-series attributes from the batch
        if dataloader_idx > 0 and not self.training:
            return {}
        tabular_attrs = {
            attr: attr_data.unsqueeze(0) if attr_data.ndim == 0 else attr_data
            for attr, attr_data in batch.items()
            if attr in self.hparams["tabular_attrs"]
        }

        time_series_attrs = {
            attr: attr_data.unsqueeze(0) if attr_data.ndim == 1 else attr_data
            for attr, attr_data in filter_time_series_attributes(
                batch, views=self.hparams.views, attrs=self.hparams.time_series_attrs
            )[0].items()
        }
        time_series_notna_mask = filter_time_series_attributes(
            batch, views=self.hparams.views, attrs=self.hparams.time_series_attrs
        )[1]
        if time_series_notna_mask is not None:
            time_series_notna_mask = (
                time_series_notna_mask.unsqueeze(0) if time_series_notna_mask.ndim == 1 else time_series_notna_mask
            )

        # print(f'time_series_attrs: {time_series_attrs}')
        # print(f"batch id: {batch['id']}")
        # print(f'time_series_notna_mask: {time_series_notna_mask}')

        if self.hparams["split_finetuning"] == 0.0 and (self.training or len(self.X_train_for_inference) == 0):
            for i in range(len(batch[self.target_attr])):
                X_batch_full, y_batch_support, y_batch_query, ts_tokens, avail_mask = self.process_data(
                    tabular_attrs, time_series_attrs, time_series_notna_mask, leave_one_out_index=i
                )  # (N, S, E), (N, S)

                metrics = {}
                losses = []
                if self.predict_losses is not None:
                    metrics.update(
                        self._prediction_shared_step(
                            X_batch_full, y_batch_support, y_batch_query, batch, batch_idx, ts_tokens, avail_mask
                        )
                    )
                    losses.append(metrics["s_loss"])

                # Compute the sum of the (weighted) losses
                metrics["loss"] = sum(losses)

        else:
            X_batch_full, y_batch_support, y_batch_query, ts_tokens, avail_mask = self.process_data(
                tabular_attrs, time_series_attrs, time_series_notna_mask
            )  # (N, S, E), (N, S)

            metrics = {}
            losses = []
            if self.predict_losses is not None:
                metrics.update(
                    self._prediction_shared_step(
                        X_batch_full, y_batch_support, y_batch_query, batch, batch_idx, ts_tokens, avail_mask
                    )
                )
                losses.append(metrics["s_loss"])

            # Compute the sum of the (weighted) losses
            metrics["loss"] = sum(losses)

        return metrics

    def _prediction_shared_step(
        self,
        X_batch_full: Tensor,
        y_batch_support: Tensor,
        y_batch_query: Tensor,
        batch: PatientDataTarget,
        batch_idx: int,
        ts_tokens: Tensor,
        avail_mask: Tensor,
    ) -> Dict[str, Tensor]:
        # Forward pass through the encoder without gradient computation to fine-tune only the prediction heads
        assert (
            self.prediction_heads is not None
        ), "You requested to perform a prediction task, but the model does not include any prediction heads."
        prediction = self.encode(X_batch_full, y_batch_support, ts_tokens, avail_mask)
        # prediction = self.encode(batch_loader, in_tokens, avail_mask)
        predictions = {}
        for attr, prediction_head in self.prediction_heads.items():
            pred = prediction_head(prediction)
            # predictions[attr] = pred.squeeze(dim=0)
            predictions[attr] = pred
            if self.time_series_processing_option == "collapse":
                # Average out the predictions across time-series attributes
                n_attrs = self.n_time_series_attrs
                predictions[attr] = predictions[attr].view(-1, n_attrs, predictions[attr].shape[-1]).mean(dim=1)

        # Compute the loss/metrics for each target attribute, ignoring items for which targets are missing
        losses, metrics = {}, {}

        # if not self.training and len(self.X_train_for_inference) > 0:
        #     target_batch = torch.cat([y_batch_support, y_batch_query], dim=0)
        # else:
        target_batch = y_batch_query

        for attr, loss in self.predict_losses.items():
            target, y_hat = target_batch, predictions[attr]

            # if attr in TabularAttribute.categorical_attrs():
            #     notna_mask = target != MISSING_CAT_ATTR
            # else:  # attr in TabularAttribute.numerical_attrs():
            #     notna_mask = ~target.isnan()

            target = target.float() if attr in TabularAttribute.binary_attrs() else target.long()

            losses[f"{loss.__class__.__name__.lower().replace('loss', '')}/{attr}"] = loss(
                y_hat,
                target,
            )
            for metric_tag, metric in self.metrics[attr].items():
                metric.update(y_hat, target)

        losses["s_loss"] = torch.stack(list(losses.values())).mean()
        metrics.update(losses)

        return metrics

    def on_test_epoch_end(self):
        all_metrics = {}
        for attr in self.predict_losses:
            for metric_tag, metric in self.metrics[attr].items():
                metrics_value = metric.compute()
                self.log(f"test_{metric_tag}/{attr}", metrics_value)
                all_metrics[f"{metric_tag}/{attr}"] = (
                    metrics_value.item() if hasattr(metrics_value, "item") else metrics_value
                )
                metric.reset()

        # Print metrics to terminal
        logger.info(f"Test metrics: {all_metrics}")

    @torch.inference_mode()
    def predict_step(self, batch: PatientData, batch_idx: int, dataloader_idx: int = 0) -> Tuple[  # noqa: D102
        Tensor,
        Optional[Dict[TabularAttribute, Tensor]],
        Optional[Dict[TabularAttribute, Tensor]],
        Optional[Dict[TabularAttribute, Tensor]],
        Optional[Dict[str, Tensor]],
        Optional[Dict[str, Tensor]],
    ]:
        # print(f"time series attrs: {time_series_attrs}")
        # print(f"batch id: {batch['id']}")
        # print(f"time series notna mask: {time_series_notna_mask}")
        tabular_attrs = {
            attr: attr_data.unsqueeze(0) if attr_data.ndim == 0 else attr_data
            for attr, attr_data in batch.items()
            if attr in self.hparams["tabular_attrs"]
        }
        time_series_attrs = {
            attr: attr_data.unsqueeze(0) if attr_data.ndim == 1 else attr_data
            for attr, attr_data in filter_time_series_attributes(
                batch, views=self.hparams.views, attrs=self.hparams.time_series_attrs
            )[0].items()
        }
        time_series_notna_mask = filter_time_series_attributes(
            batch, views=self.hparams.views, attrs=self.hparams.time_series_attrs
        )[1]
        if time_series_notna_mask is not None:
            time_series_notna_mask = (
                time_series_notna_mask.unsqueeze(0) if time_series_notna_mask.ndim == 1 else time_series_notna_mask
            )
        # Encoder's output
        out_features = self(tabular_attrs, time_series_attrs, time_series_notna_mask)

        # Remove unnecessary batch dimension from the different outputs
        # (only do this once all downstream inferences have been performed)
        out_features = out_features.squeeze(dim=0)
        if self.time_series_processing_option == "collapse":
            # Average out the latent vectors across time-series attributes
            n_attrs = self.n_time_series_attrs
            out_features = out_features.view(-1, n_attrs, out_features.shape[-1]).mean(dim=1)

        # If the model has targets to predict, output the predictions
        predictions = None
        if self.prediction_heads:
            predictions = self(tabular_attrs, time_series_attrs, time_series_notna_mask, task="predict")

        if predictions is not None:
            predictions = {
                attr: prediction.unsqueeze(dim=0) if prediction.ndim == 1 else prediction
                for attr, prediction in predictions.items()
            }

        if self.hparams["latent_representation"]:
            latent_ts, latent_unique, latent_shared = self.get_latent_vectors(
                batch, batch_idx, tabular_attrs, time_series_attrs
            )
            latent_ts = latent_ts.squeeze(dim=0)
            latent_unique = latent_unique.squeeze(dim=0)
            if latent_shared is not None:
                latent_shared = latent_shared.squeeze(dim=0)
                latent_dict = {
                    "time-series": latent_ts,
                    "tabular specific": latent_unique,
                    "tabular shared": latent_shared,
                }
            else:
                latent_dict = {
                    "time-series": latent_ts,
                    "tabular": latent_unique,
                }
        else:
            latent_dict = None
        if self.hparams["explainability"]:
            all_tokens = self(tabular_attrs, time_series_attrs)
            vectors_dict = {f"tab_shared_{i}": all_tokens[:, i, :] for i in range(13)}
            vectors_dict.update({f"ts_{i}": all_tokens[:, i + 13, :] for i in range(14)})
            vectors_dict.update({f"tab_specific_{i}": all_tokens[:, i + 13 + 14, :] for i in range(13)})
        else:
            vectors_dict = None

        # Squeeze pred before returning
        if predictions is not None:
            predictions = {attr: prediction.squeeze(dim=0) for attr, prediction in predictions.items()}
            if self.time_series_processing_option == "collapse":
                # Average out the predictions across time-series attributes
                n_attrs = self.n_time_series_attrs
                predictions = {
                    attr: prediction.view(-1, n_attrs, prediction.shape[-1]).mean(dim=1)
                    for attr, prediction in predictions.items()
                }
        return out_features, predictions, latent_dict, vectors_dict
