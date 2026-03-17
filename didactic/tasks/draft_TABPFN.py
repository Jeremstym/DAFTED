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
from didactic.models.time_series import TimeSeriesEmbedding

from tabpfn.preprocessing import EnsembleConfig, PreprocessorConfig
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator

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
        n_estimators: int,
        n_classes: int,
        add_fingerprint_feature: bool,
        feature_shift_decoder: Literal["shuffle", "rotate"],
        polynomial_features: Literal["all", "no"],
        subsample_size: int | None = None,
        class_shift_method: Optional[Literal["shuffle", "rotate"]] | None = None,
        views: Sequence[ViewEnum] = tuple(ViewEnum),
        predict_losses: Optional[Dict[TabularAttribute | str, Callable[[Tensor, Tensor], Tensor]] | DictConfig] = None,
        tabular_tokenizer: Optional[TabularEmbedding | DictConfig] = None,
        time_series_tokenizer: Optional[TimeSeriesEmbedding | DictConfig] = None,
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

        # if contrastive_loss is None and predict_losses is None:
        #     raise ValueError(
        #         "You should provide at least one of  `contrastive_loss` or `predict_losses`. Providing only "
        #         "`contrastive_loss` will run a self-supervised (pre)training phase. Providing only `predict_losses` "
        #         "will run a fully-supervised training phase. Finally, providing both at the same time will train the "
        #         "model in fully-supervised mode, with the self-supervised loss as an auxiliary term."
        #     )

        if not tabular_tokenizer and tabular_attrs:
            raise ValueError(
                f"You have requested the following tabular attributes: "
                f"{[str(attr) for attr in tabular_attrs]}, but have not configured a tokenizer for tabular attributes. "
                f"Either provide this tokenizer (through the `tabular_tokenizer` parameter) or remove any tabular "
                f"attributes (by setting the `tabular_attrs` to be an empty list)."
            )
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
            attr for attr in self.hparams["tabular_attrs"] if attr in TabularAttribute.categorical_attrs()
        ]
        self.tabular_cat_attrs_cardinalities = [
            len(TABULAR_CAT_ATTR_LABELS[cat_attr]) for cat_attr in self.tabular_cat_attrs
        ]

        self.cat_idxs = []
        for attr in self.tabular_cat_attrs:
            self.cat_idxs.append(self.hparams["tabular_attrs"].index(attr))

        self.target_attr = target_attr

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
                        # "auprc": functools.partial(average_precision, task="binary"),
                        # "f1": functools.partial(f1_score, task="binary"),
                    ],
                )
            else:  # attr in TabularAttribute.categorical_attrs()
                num_classes = len(TABULAR_CAT_ATTR_LABELS[attr])
                self.metrics[attr] = MetricCollection(
                    [
                        MulticlassAccuracy(num_classes=num_classes, average="micro"),
                        MulticlassAUROC(num_classes=num_classes, average="none"),
                        MulticlassAveragePrecision(num_classes=num_classes, average="none"),
                        MulticlassF1Score(num_classes=num_classes, average="none"),
                    ]
                )

        # Compute shapes relevant for defining the models' architectures
        self.n_tabular_attrs = len(self.hparams["tabular_attrs"])
        self.n_time_series_attrs = len(self.hparams["time_series_attrs"]) * len(self.hparams["views"])
        self.sequence_length = self.n_time_series_attrs + self.n_tabular_attrs

        # Self-supervised losses and metrics
        # Initialize transformer encoder and self-supervised + prediction heads
        self.encoder, self.contrastive_head, self.prediction_heads = self.configure_model()

        # self.encoder._initialize_model_variables()

        # Configure tokenizers and extract relevant info about the models' architectures
        self.nhead, self.separate_modality = check_model_encoder(self.encoder, self.hparams)

        # Initialize the tabular and time-series tokenizers
        # self.tabular_tokenizer: TabularEmbedding | None
        # if tabular_attrs:
        #     if isinstance(tabular_tokenizer, DictConfig):
        #         tabular_tokenizer_instance = hydra.utils.instantiate(
        #             tabular_tokenizer,
        #             n_num_features=len(self.tabular_num_attrs),
        #             cat_cardinalities=self.tabular_cat_attrs_cardinalities,
        #         )
        # else:
        #     # Set tokenizer to `None` if it's not going to be used
        #     tabular_tokenizer_instance = None
        # self.tabular_tokenizer = tabular_tokenizer_instance

        self.time_series_tokenizer: TimeSeriesEmbedding | None
        if time_series_attrs:
            if isinstance(time_series_tokenizer, DictConfig):
                time_series_tokenizer_instance = hydra.utils.instantiate(time_series_tokenizer)
        else:
            # Set tokenizer to `None` if it's not going to be used
            time_series_tokenizer_instance = None
        self.time_series_tokenizer = time_series_tokenizer_instance

        self.mask_token = None

        # if perform_lora:
        #     adapter_encoder: nn.Module
        #     lora_linar = LoRALinear
        #     if not self.separate_modality:
        #         adapter_encoder = AdapterWrapperFT_Transformer(self.encoder, lora_linar, gamma=8, lora_alpha=8)
        #     elif isinstance(self.encoder, didactic.models.transformer.FT_Interleaved):
        #         adapter_encoder = AdapterWrapperFT_Interleaved(self.encoder, lora_linar, gamma=8, lora_alpha=8)
        #     else:
        #         adapter_encoder = AdapterWrapperFT_Transformer_CrossAtt(self.encoder, lora_linar, gamma=8, lora_alpha=8)
        #     setattr(self, "encoder", adapter_encoder)

        # preprocess_transforms = [PreprocessorConfig("none", differentiable=True)]

        # self.ensemble_configs = functools.partial(
        #     EnsembleConfig.generate_for_classification,
        #     n=n_estimators,
        #     add_fingerprint_feature=add_fingerprint_feature,
        #     feature_shift_decoder=feature_shift_decoder,
        #     polynomial_features=polynomial_features,
        #     preprocessor_configs=cast("Sequence[PreprocessorConfig]", preprocess_transforms),
        #     subsample_size=None,
        #     class_shift_method=None,
        #     n_classes=n_classes,
        #     random_state=self.hparams["seed"],
        # )

        self.X_train_for_inference = torch.Tensor()
        self.y_train_for_inference = torch.Tensor()

    @property
    def example_input_array(
        self,
    ) -> Tuple[Dict[TabularAttribute, Tensor], Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor]]:
        """Redefine example input array based on the cardiac attributes provided to the model."""
        # 2 is the size of the batch in the example
        tab_attrs = {attr: torch.randn(10) for attr in self.tabular_num_attrs}
        # Only generate 0/1 labels, to avoid generating labels bigger than the number of classes, which would lead to
        # an index out of range error when looking up the embedding of the class in the categorical feature tokenizer
        tab_attrs.update({attr: torch.randint(10, (10,)) for attr in self.tabular_cat_attrs})
        labels = torch.randperm(5)
        tab_attrs[self.target_attr] = torch.cat([labels, labels])  # Double the labels to match batch size of 10
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
        encoder = hydra.utils.instantiate(self.hparams["model"]["encoder"])

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

        return encoder, contrastive_head, prediction_heads

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizer to ignore parameters that should remain frozen (e.g. tokenizers)."""
        # Set requires_grad to True for model
        # for param in self.encoder.model_.parameters():
        #     param.requires_grad = True
        # params = list(filter(lambda p: p.requires_grad, self.encoder.model_.parameters()))
        return super().configure_optimizers()

    @auto_move_data
    def process_data(
        self,
        tabular_attrs: Dict[TabularAttribute, Tensor],
        time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor],
        time_series_notna_mask: Optional[Tensor] = None,
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

        # Tokenize the attributes
        if time_series_attrs:
            assert (
                self.time_series_tokenizer is not None
            ), "Time-series tokenizer must be configured to tokenize time-series."
            time_series_attrs_tokens = self.time_series_tokenizer(time_series_attrs)  # S * (N, ?) -> (N, S_ts, E)
            ts_tokens_list.extend(time_series_attrs_tokens)

            # Indicate that, when time-series tokens are requested, they are always available
            if time_series_notna_mask is None:
                time_series_notna_mask = torch.full(
                    time_series_attrs_tokens.shape[:2], True, device=time_series_attrs_tokens.device
                )

            # time_series_notna_mask = torch.full(
            #     time_series_attrs_tokens.shape[:2], True, device=time_series_attrs_tokens.device
            # )
            notna_mask_list.extend(time_series_notna_mask)

        if tabular_attrs:
            # assert self.tabular_tokenizer is not None, "Tabular tokenizer must be configured to tokenize tabular data."
            # tab_attrs_tokens, tab_notna_mask = self.tabular_tokenizer(
            #     tabular_attrs=tabular_attrs,
            #     tabular_num_attrs=self.tabular_num_attrs,
            #     tabular_cat_attrs=self.tabular_cat_attrs,
            # )

            y = tabular_attrs.pop(self.target_attr)
            label = y.clone()
            X = torch.hstack([v.unsqueeze(1) for v in tabular_attrs.values()])

            if self.training:
                self.X_train_for_inference = self.X_train_for_inference.to(device=X.device)
                self.y_train_for_inference = self.y_train_for_inference.to(device=y.device)
                self.X_train_for_inference = torch.cat([self.X_train_for_inference, X], dim=0)
                self.y_train_for_inference = torch.cat([self.y_train_for_inference, y], dim=0)

            indices = list(range(len(y)))
            try:
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=self.hparams["split_finetuning"],
                    random_state=self.hparams["seed"],
                    stratify=label.cpu().numpy(),
                )
            except ValueError:
                print("Stratified splitting failed, performing non-stratified split instead.")
                train_indices, test_indices = train_test_split(
                    indices,
                    test_size=self.hparams["split_finetuning"],
                    random_state=self.hparams["seed"],
                )

            # cat_idxs = []
            # if self.tabular_cat_attrs:
            #     for attr in self.tabular_cat_attrs:
            #         cat_idxs.append(self.hparams["tabular_attrs"].index(attr))
            # cat_idxs = [torch.tensor(cat_idxs, device=ts_tokens.device)]
            # self.cat_idxs = [cat_idxs]
            # self.encoder.categorical_features_indices = self.cat_idxs

            # batch_dataset = self.encoder.get_preprocessed_datasets(X, y, splitter, len(y))

            # batch_loader = DataLoader(
            #     batch_dataset,
            #     batch_size=1,
            #     collate_fn=meta_dataset_collator,
            # )

            X_batch_support = torch.as_tensor(X[train_indices], dtype=torch.float32)
            X_batch_query = torch.as_tensor(X[test_indices], dtype=torch.float32)
            y_batch_support = torch.as_tensor(y[train_indices], dtype=torch.float32)
            y_batch_query = torch.as_tensor(y[test_indices], dtype=torch.float32)

            X_batch_full = torch.cat([X_batch_support, X_batch_query], dim=0).unsqueeze(1)

            # tokens_list.append(tab_attrs_tokens)
            # notna_mask_list.extend(tab_notna_mask)

        # Cast to float to make sure tokens are not represented using double
        # ts_tokens = torch.cat(ts_tokens_list, dim=1).float()  # (N, S_ts + S_tab, E)
        ts_tokens = torch.Tensor(ts_tokens_list).float()  # (N, S_ts, E)
        # Cast to bool to make sure attention mask is represented by bool
        # notna_mask = torch.cat(notna_mask_list, dim=1).bool()  # (N, S_ts + S_tab)
        notna_mask = torch.Tensor(notna_mask_list).bool()  # (N, S_ts)

        return X_batch_full, y_batch_support, y_batch_query, ts_tokens, notna_mask
        # return (batch_loader if tabular_attrs else None), ts_tokens, notna_mask

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
        # y_batch_query: Tensor,
        # batch_loader: Dataset,
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

        ts_tokens = self.preprocess_tokens(ts_tokens, avail_mask, enable_augments=enable_augments)

        # for X_batch_support, X_batch_query, y_batch_support, y_batch_query, cat_ixs, confs in batch_loader:

        # X_batch_support = [Xb.unsqueeze(1) for Xb in X_batch_support]
        # X_batch_query = [Xb.unsqueeze(1) for Xb in X_batch_query]
        # y_batch_support = [yb.unsqueeze(1) for yb in y_batch_support]

        # for list_ix in cat_ixs:
        #     list_ix.extend(self.cat_idxs)

        # Set inputs on device
        # X_batch_support = [Xb.to(device="cuda:0") for Xb in X_batch_support]
        # X_batch_query = [Xb.to(device="cuda:0") for Xb in X_batch_query]
        # y_batch_support = [yb.to(device="cuda:0") for yb in y_batch_support]

        # self.encoder.fit_from_preprocessed(X_batch_support, y_batch_support, cat_ixs, confs)
        # predictions = self.encoder.forward(X_batch_query, return_logits=True)

        # ensemble_cfgs = self.ensemble_configs(max_index=len(X_batch_support))
        # cat_indices = [self.cat_idxs] * self.hparams["n_estimators"]
        # conf_list = [ensemble_cfgs[0]]  # * self.hparams["n_estimators"]

        # self.encoder.fit_from_preprocessed(X_batch_support, y_batch_support, cat_indices, ensemble_cfgs)
        if not self.training and len(self.X_train_for_inference) > 0:
            # X_test = torch.cat([X_batch_support, X_batch_query], dim=0)
            # self.encoder.fit_from_preprocessed(
            #     [self.X_train_for_inference.unsqueeze(0)],
            #     [self.y_train_for_inference.unsqueeze(0)],
            #     [cat_indices],
            #     [conf_list],
            # )
            # out_features = self.encoder._raw_predict(X_test, return_logits=True)
            # prediction = self.encoder.forward([X_test.unsqueeze(0)], return_logits=True)
            X_full_train_test = torch.cat([self.X_train_for_inference.unsqueeze(1), X_batch_full], dim=0)
            y_train = self.y_train_for_inference
            out_features = self.encoder(X_full_train_test, y_train)
        else:
            # self.encoder.fit_from_preprocessed(
            #     [X_batch_support.unsqueeze(0)], [y_batch_support.unsqueeze(0)], [cat_indices], [conf_list]
            # )

            out_features = self.encoder(X_batch_full, y_batch_support)

            # prediction = self.encoder.forward([X_batch_query.unsqueeze(0)], return_logits=True)
            # out_features = self.encoder._raw_predict(X_batch_query, return_logits=True)

        # out_tokens = self.encoder(tokens)

        # out_features = out_tokens.squeeze(1)  # (N, 1, E) -> (N, E)

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

        X_batch_full, y_batch_support, y_batch_query, ts_tokens, avail_mask = self.process_data(
            tabular_attrs, time_series_attrs, time_series_notna_mask
        )  # (N, S, E), (N, S)
        # batch_loader, ts_tokens, avail_mask = self.process_data(
        #     tabular_attrs, time_series_attrs, time_series_notna_mask
        # )  # (N, S, E), (N, S)
        out_features = self.encode(X_batch_full, y_batch_support, ts_tokens, avail_mask)  # (N, S, E) -> (N, E)
        # out_features = self.encode(batch_loader, ts_tokens, avail_mask)  # (N, S, E) -> (N, E)

        # Early return if requested task requires no prediction heads
        if task == "encode":
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
        X_batch_support, X_batch_query, y_batch_support, y_batch_query, ts_tokens, avail_mask = self.process_data(
            tabular_attrs, time_series_attrs, None
        )  # (N, S, E), (N, S)
        return self.encode(X_batch_support, X_batch_query, y_batch_support, y_batch_query, ts_tokens, avail_mask)

    def evaluate_model(
        self,
        # X_train: np.ndarray,
        # y_train: np.ndarray,
        # X_test: np.ndarray,
        # y_test: np.ndarray,
        tabular_attrs: Dict[TabularAttribute, Tensor],
    ) -> tuple[float, float]:
        """Evaluates the model's performance on the test set."""

        y_test = tabular_attrs.pop(self.target_attr).cpu().numpy()
        X_test = torch.hstack([v.unsqueeze(1) for v in tabular_attrs.values()]).cpu().numpy()

        self.encoder.fit(self.X_train_for_inference.cpu().numpy(), self.y_train_for_inference.cpu().numpy())

        probabilities = self.encoder.predict_proba(X_test)

        # accuracy = accuracy_score(y_test, np.argmax(probabilities, axis=1))
        # auroc = roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro")
        # auprc = average_precision_score(y_test, probabilities, average="macro", multi_class="ovr")
        # f1 = f1_score(y_test, np.argmax(probabilities, axis=1), average="macro")
        # log_loss_score = log_loss(y_test, probabilities)

        metrics, loss = {}, {}
        for attr in self.predict_losses:
            for metric_tag, metric in self.metrics[attr].items():
                metrics[f"{metric_tag}/{attr}"] = metric(
                    torch.tensor(probabilities, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.long),
                )
            loss[
                f"{self.predict_losses[attr].__class__.__name__.lower().replace('loss', '')}/{attr}"
            ] = self.predict_losses[attr](
                torch.tensor(probabilities, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            )
        loss["s_loss"] = torch.stack(list(loss.values())).mean()
        metrics.update(loss)

        return metrics

    def _shared_step(self, batch: PatientDataTarget, batch_idx: int) -> Dict[str, Tensor]:
        # Extract tabular and time-series attributes from the batch
        tabular_attrs = {attr: attr_data for attr, attr_data in batch.items() if attr in self.hparams["tabular_attrs"]}
        time_series_attrs, time_series_notna_mask = filter_time_series_attributes(
            batch, views=self.hparams["views"], attrs=self.hparams["time_series_attrs"]
        )
        # print(f'time_series_attrs: {time_series_attrs}')
        # print(f"batch id: {batch['id']}")
        # print(f'time_series_notna_mask: {time_series_notna_mask}')

        X_batch_full, y_batch_support, y_batch_query, ts_tokens, avail_mask = self.process_data(
            tabular_attrs, time_series_attrs, time_series_notna_mask
        )  # (N, S, E), (N, S)
        # batch_loader, ts_tokens, avail_mask = self.process_data(
        #     tabular_attrs, time_series_attrs, time_series_notna_mask
        # )  # (N, S, E), (N, S)

        metrics = {}
        losses = []
        if self.predict_losses is not None:  # run fully-supervised prediction step
            if not self.training and len(self.X_train_for_inference) > 0:
                metrics.update(
                    self._prediction_shared_step(
                        X_batch_full,
                        y_batch_support,
                        y_batch_query,
                        batch,
                        batch_idx,
                        ts_tokens,
                        avail_mask,
                    )
                    # self.evaluate_model(tabular_attrs)
                )
            else:
                metrics.update(
                    self._prediction_shared_step(
                        X_batch_full,
                        y_batch_support,
                        y_batch_query,
                        batch,
                        batch_idx,
                        ts_tokens,
                        avail_mask,
                    )
                    # self._prediction_shared_step(batch_loader, batch, batch_idx, ts_tokens, avail_mask)
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
        # batch_loader: DataLoader,
        batch: PatientDataTarget,
        batch_idx: int,
        in_tokens: Tensor,
        avail_mask: Tensor,
    ) -> Dict[str, Tensor]:
        # Forward pass through the encoder without gradient computation to fine-tune only the prediction heads
        assert (
            self.prediction_heads is not None
        ), "You requested to perform a prediction task, but the model does not include any prediction heads."
        prediction = self.encode(X_batch_full, y_batch_support, in_tokens, avail_mask)
        # prediction = self.encode(batch_loader, in_tokens, avail_mask)
        predictions = {}
        for attr, prediction_head in self.prediction_heads.items():
            pred = prediction_head(prediction)
            predictions[attr] = pred.squeeze(dim=0)
        # Print patient id related to predictions
        # print(f'Patient ID: {batch["id"]}, Predictions: {predictions}')
        # Compute the loss/metrics for each target attribute, ignoring items for which targets are missing
        losses, metrics = {}, {}

        # for X_batch_support, X_batch_query, y_batch_support, y_batch_query, _, _ in batch_loader:

        if not self.training and len(self.X_train_for_inference) > 0:
            target_batch = torch.cat([y_batch_support, y_batch_query], dim=0)
        else:
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

            # for metric_tag, metric in self.metrics[attr].items():
            #     metrics[f"{metric_tag}/{attr}"] = metric(y_hat[notna_mask], target[notna_mask])
            for metric_tag, metric in self.metrics[attr].items():
                metric_res = metric(y_hat, target)

                # For multiclass categorical attributes, metrics are not averaged by default, so log them for each
                # class separately and then average them manually
                if attr in TabularAttribute.categorical_attrs():
                    # for class_label, metric_res_for_class in zip(TABULAR_CAT_ATTR_LABELS[attr], metric_res):
                    #     metrics[f"{metric_tag}/{attr}/{class_label}"] = metric_res_for_class
                    metric_res = metric_res.mean()

                metrics[f"{metric_tag}/{attr}"] = metric_res

        # Reduce loss across the multiple targets
        if self.hparams["dummy_mode"]:
            losses["s_loss"] = 0.0
        else:
            losses["s_loss"] = torch.stack(list(losses.values())).mean()
        metrics.update(losses)

        return metrics

    @torch.inference_mode()
    def predict_step(self, batch: PatientData, batch_idx: int, dataloader_idx: int = 0) -> Tuple[  # noqa: D102
        Tensor,
        Optional[Dict[TabularAttribute, Tensor]],
        Optional[Dict[TabularAttribute, Tensor]],
        Optional[Dict[TabularAttribute, Tensor]],
        Optional[Dict[str, Tensor]],
        Optional[Dict[str, Tensor]],
    ]:
        # Extract tabular and time-series attributes from the patient and add batch dimension
        # tabular_attrs = {attr: attr_data.unsqueeze(dim=0) for attr, attr_data in batch.items() if attr in self.hparams["tabular_attrs"]}
        tabular_attrs = {attr: attr_data for attr, attr_data in batch.items() if attr in self.hparams["tabular_attrs"]}
        time_series_attrs, time_series_notna_mask = filter_time_series_attributes(
            batch, views=self.hparams["views"], attrs=self.hparams["time_series_attrs"]
        )
        # Add batch dimension to the time-series attributes
        # time_series_attrs = {
        #     (view, attr): attr_data.unsqueeze(dim=0)
        #     for (view, attr), attr_data in time_series_attrs.items()
        # }
        time_series_attrs = {(view, attr): attr_data for (view, attr), attr_data in time_series_attrs.items()}
        # Add batch dimension to the time-series notna mask
        if time_series_notna_mask is not None and time_series_notna_mask.ndim == 1:
            time_series_notna_mask = time_series_notna_mask.unsqueeze(dim=0)

        # print(f"time series attrs: {time_series_attrs}")
        # print(f"batch id: {batch['id']}")
        # print(f"time series notna mask: {time_series_notna_mask}")

        # Encoder's output
        out_features = self(tabular_attrs, time_series_attrs, time_series_notna_mask)

        # If the model has targets to predict, output the predictions
        predictions = None
        if self.prediction_heads:
            predictions = self(tabular_attrs, time_series_attrs, time_series_notna_mask, task="predict")

        # Remove unnecessary batch dimension from the different outputs
        # (only do this once all downstream inferences have been performed)
        out_features = out_features.squeeze(dim=0)
        if predictions is not None:
            # Associate each prediction to the corresponding batch idx
            # predictions: (batch_size, ...)
            # batch["id"]: (batch_size,)

            # If predictions is a tensor, convert to list if needed
            predictions = {attr: prediction.squeeze(dim=0) for attr, prediction in predictions.items()}
            pred_list = list(predictions.values())[0]
            # Associate each prediction to its id
            pred_id_map = {id_: pred for id_, pred in zip(batch["id"], pred_list)}

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
        # print(f"Patient ID: {batch['id']}, Predictions: {predictions}, ")
        return out_features, pred_id_map, latent_dict, vectors_dict
