import functools
import importlib
import itertools
import logging
import math
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, cast

import hydra
import torch
from dataprocessing.data.augmentation.base import mask_tokens, random_masking
from dataprocessing.data.cardinal.config import CardinalTag, TabularAttribute, TimeSeriesAttribute
from dataprocessing.data.cardinal.config import View as ViewEnum
from dataprocessing.data.cardinal.datapipes import MISSING_CAT_ATTR, PatientData, PatientDataTarget, PatientDataInference, filter_time_series_attributes
from dataprocessing.data.cardinal.utils.attributes import TABULAR_CAT_ATTR_LABELS
from dataprocessing.tasks.generic import SharedStepsTask
from dataprocessing.utils.decorators import auto_move_data
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig

# import dataprocessing
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict, init
from torchmetrics.functional import accuracy, auroc, average_precision, f1_score, mean_absolute_error
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryAveragePrecision, MulticlassAccuracy, MulticlassAUROC, MulticlassAveragePrecision, MulticlassF1Score
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics import MetricCollection

from dafted.utils.compliance import check_model_encoder
import dafted.models.transformer
from dafted.models.adaptater import (
    AdapterWrapperFT_Interleaved,
    AdapterWrapperFT_Transformer,
    AdapterWrapperFT_Transformer_CrossAtt,
    LoRALinear,
)
from dafted.models.layers import CLSToken, PositionalEncoding, SequencePooling
from dafted.models.losses import ReconstructionLoss
from dafted.models.tabular import TabularEmbedding
from dafted.models.time_series import TimeSeriesEmbedding

logger = logging.getLogger(__name__)

torch.set_printoptions(threshold=10000)


class CardiacMultimodalRepresentationTask(SharedStepsTask):
    """Multi-modal transformer to learn a representation from cardiac imaging and patient records data."""

    def __init__(
        self,
        embed_dim: int,
        tabular_attrs: Sequence[TabularAttribute | str],
        time_series_attrs: Sequence[TimeSeriesAttribute],
        views: Sequence[ViewEnum] = tuple(ViewEnum),
        predict_losses: Optional[Dict[TabularAttribute | str, Callable[[Tensor, Tensor], Tensor]] | DictConfig] = None,
        ordinal_mode: bool = True,
        contrastive_loss: Optional[Callable[[Tensor, Tensor, Tensor], Tensor] | DictConfig] = None,
        contrastive_loss_weight: float = 0.0,
        orthogonal_loss: Optional[Callable[[Tensor, Tensor], Tensor] | DictConfig] = None,
        orthogonal_loss_weight: float = 0.0,
        reconstruction_loss: Optional[ReconstructionLoss | DictConfig] = None,
        reconstruction_loss_weight: float = 0.0,
        inter_sample_loss: Optional[Callable[[Tensor, Tensor], Tensor] | DictConfig] = None,
        inter_sample_loss_weight: float = 0.0,
        attribute_reg_loss: Optional[Callable[[Tensor, Dict[TabularAttribute, Tensor]], Tensor] | DictConfig] = None,
        attribute_reg_loss_weight: float = 0.0,
        tabular_tokenizer: Optional[TabularEmbedding | DictConfig] = None,
        time_series_tokenizer: Optional[TimeSeriesEmbedding | DictConfig] = None,
        cls_token: bool = True,
        mtr_p: float | Tuple[float, ...] = 0.0,
        mt_by_attr: bool = False,
        perform_lora: bool = False,
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
            ordinal_mode: Whether to consider applicable targets as ordinal variables, which means:
                - Applying a constraint to enforce an unimodal softmax output from the prediction heads;
                - Predicting a new continuum value for each ordinal target, namely the param. of the unimodal softmax.
            contrastive_loss: Self-supervised criterion to use as contrastive loss between pairs of (N, E) collections
                of feature vectors, in a contrastive learning step that follows the SCARF pretraining.
                (see ref: https://arxiv.org/abs/2106.15147)
            contrastive_loss_weight: Factor by which to weight the `contrastive_loss` in the overall loss.
            tabular_tokenizer: Tokenizer that can process tabular, i.e. patient records, data.
            time_series_tokenizer: Tokenizer that can process time-series data.
            cross_attention_module: Module to use for cross-attention between the tabular and time-series tokens.
            cls_token: If `True`, adds a CLS token to use as the encoder's output token.
                If `False`, the output token is obtained by sequence pooling over the encoder's output tokens.
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

        if contrastive_loss is None and predict_losses is None:
            raise ValueError(
                "You should provide at least one of  `contrastive_loss` or `predict_losses`. Providing only "
                "`contrastive_loss` will run a self-supervised (pre)training phase. Providing only `predict_losses` "
                "will run a fully-supervised training phase. Finally, providing both at the same time will train the "
                "model in fully-supervised mode, with the self-supervised loss as an auxiliary term."
            )

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
        if cls_token:
            self.token_tags = self.token_tags + ("CLS",)

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
                TabularAttribute[attr]: ( # type: ignore[misc]
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
                # nn.ModuleDict(
                #     {
                #         "acc": MulticlassAccuracy(num_classes=num_classes, average="none"),
                #         "auroc": MulticlassAUROC(num_classes=num_classes, average="none"),
                #         "auprc": MulticlassAveragePrecision(num_classes=num_classes, average="none"),
                #         "f1": MulticlassF1Score(num_classes=num_classes, average="none"),
                #         "auprc": functools.partial(
                #         average_precision, task="multiclass", num_classes=num_classes, average="macro"
                #         ),
                #         "f1": functools.partial(f1_score, task="multiclass", num_classes=num_classes, average="macro"),
                #     }
                # )
        # Switch on ordinal mode if i) it's enabled, and ii) there are ordinal targets to predict
        self.hparams["ordinal_mode"] = self.hparams["ordinal_mode"] and any(
            attr in TabularAttribute.ordinal_attrs() for attr in self.predict_losses
        )

        # Compute shapes relevant for defining the models' architectures
        self.n_tabular_attrs = len(self.hparams["tabular_attrs"])
        self.n_time_series_attrs = len(self.hparams["time_series_attrs"]) * len(self.hparams["views"])
        if self.hparams["ts_pooling_factor"]:
            self.n_time_series_attrs *= self.hparams["ts_pooling_factor"]
        self.sequence_length = self.n_time_series_attrs + self.n_tabular_attrs + self.hparams["cls_token"]

        # Self-supervised losses and metrics
        self.contrastive_loss: Optional[Callable[..., Any]] = None
        self.orthogonal_loss: Optional[Callable[..., Any]] = None
        self.reconstruction_loss_unique: Optional[Callable[..., Any]] = None
        self.inter_sample_loss: Optional[Callable[..., Any]] = None
        self.attribute_reg_loss: Optional[Callable[..., Any]] = None
        if contrastive_loss and self.hparams["contrastive_loss_weight"]:
            self.contrastive_loss = (
                hydra.utils.instantiate(contrastive_loss)
                if isinstance(contrastive_loss, DictConfig)
                else contrastive_loss
            )
        if orthogonal_loss and self.hparams["orthogonal_loss_weight"]:
            self.orthogonal_loss = (
                hydra.utils.instantiate(orthogonal_loss) if isinstance(orthogonal_loss, DictConfig) else orthogonal_loss
            )

        if reconstruction_loss and self.hparams["reconstruction_loss_weight"]:
            self.reconstruction_loss_unique = ReconstructionLoss(
                num_con=len(self.tabular_num_attrs),
                cat_lengths_tabular=self.tabular_cat_attrs_cardinalities,
                d_token=self.hparams["embed_dim"],
            )
        if inter_sample_loss and self.hparams["inter_sample_loss_weight"]:
            self.inter_sample_loss = (
                hydra.utils.instantiate(inter_sample_loss)
                if isinstance(inter_sample_loss, DictConfig)
                else inter_sample_loss
            )

        if attribute_reg_loss and self.hparams["attribute_reg_loss_weight"]:
            self.attribute_reg_loss = (
                hydra.utils.instantiate(attribute_reg_loss)
                if isinstance(attribute_reg_loss, DictConfig)
                else attribute_reg_loss
            )

        # Initialize transformer encoder and self-supervised + prediction heads
        self.encoder, self.contrastive_head, self.prediction_heads = self.configure_model()

        # Configure tokenizers and extract relevant info about the models' architectures
        self.nhead, self.separate_modality = check_model_encoder(self.encoder, self.hparams)

        # Initialize the tabular and time-series tokenizers
        self.tabular_tokenizer: TabularEmbedding | None
        if tabular_attrs:
            if isinstance(tabular_tokenizer, DictConfig):
                tabular_tokenizer_instance = hydra.utils.instantiate(
                    tabular_tokenizer,
                    n_num_features=len(self.tabular_num_attrs),
                    cat_cardinalities=self.tabular_cat_attrs_cardinalities,
                )
        else:
            # Set tokenizer to `None` if it's not going to be used
            tabular_tokenizer_instance = None
        self.tabular_tokenizer = tabular_tokenizer_instance

        self.time_series_tokenizer: TimeSeriesEmbedding | None
        if time_series_attrs:
            if isinstance(time_series_tokenizer, DictConfig):
                time_series_tokenizer_instance = hydra.utils.instantiate(time_series_tokenizer)
        else:
            # Set tokenizer to `None` if it's not going to be used
            time_series_tokenizer_instance = None
        self.time_series_tokenizer = time_series_tokenizer_instance

        if self.hparams["positional_encoding"]:
            self.positional_encoding = hydra.utils.instantiate(self.hparams["positional_encoding"])

        # Initialize parameters of method for reducing the dimensionality of the encoder's output to only one token
        if self.hparams["cls_token"]:
            self.cls_token = CLSToken(self.hparams["embed_dim"])
            if self.hparams["ts_cls_token"]:
                self.cls_ts_token = CLSToken(self.hparams["embed_dim"])
        elif self.hparams["sequence_pooling"]:
            self.sequence_pooling = SequencePooling(self.hparams["embed_dim"])

        if self.hparams["mtr_p"]:
            # Initialize the mask token, which will be used to replace tokens in the input sequence
            self.mask_token: Parameter | ParameterDict
            def _init_mask_token() -> Parameter:
                # Initialize parameters of MTR's mask token
                # Since MTR is built on top of FTT and copies implementation details (including the initialization)
                # we also default to re-using their initialization
                mask_token = Parameter(torch.empty(self.hparams["embed_dim"]))
                d_sqrt_inv = 1 / math.sqrt(self.hparams["embed_dim"])
                init.uniform_(mask_token, a=-d_sqrt_inv, b=d_sqrt_inv)
                return mask_token

            if self.hparams["mt_by_attr"]:
                # Init one MASK token for each attribute
                attr_tags = self.token_tags
                if self.hparams["cls_token"]:
                    attr_tags = attr_tags[:-1]
                self.mask_token = nn.ParameterDict({attr: _init_mask_token() for attr in attr_tags})
            else:
                # Init a single universal MASK token
                self.mask_token = _init_mask_token()

        if perform_lora:
            adapter_encoder: nn.Module
            lora_linar = LoRALinear
            if not self.separate_modality:
                adapter_encoder = AdapterWrapperFT_Transformer(self.encoder, lora_linar, gamma=8, lora_alpha=8)
            elif isinstance(self.encoder, dafted.models.transformer.FT_Interleaved):
                adapter_encoder = AdapterWrapperFT_Interleaved(self.encoder, lora_linar, gamma=8, lora_alpha=8)
            else:
                adapter_encoder = AdapterWrapperFT_Transformer_CrossAtt(self.encoder, lora_linar, gamma=8, lora_alpha=8)
            setattr(self, "encoder", adapter_encoder)

        # tab_attrs_example = {attr: torch.randn(2) for attr in self.tabular_num_attrs}
        # # Only generate 0/1 labels, to avoid generating labels bigger than the number of classes, which would lead to
        # # an index out of range error when looking up the embedding of the class in the categorical feature tokenizer
        # tab_attrs_example.update({attr: torch.randint(2, (2,)) for attr in self.tabular_cat_attrs})
        # time_series_attrs_example = {
        #     (view, attr): torch.randn(2, self.hparams["data_params"].in_shape[CardinalTag.time_series_attrs][1])
        #     for view, attr in itertools.product(self.hparams["views"], self.hparams["time_series_attrs"])
        # }

        # self.example_input_array = (tab_attrs_example, time_series_attrs_example)

    @property
    def example_input_array(
        self,
    ) -> Tuple[Dict[TabularAttribute, Tensor], Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor]]:
        """Redefine example input array based on the cardiac attributes provided to the model."""
        # 2 is the size of the batch in the example
        tab_attrs = {attr: torch.randn(2) for attr in self.tabular_num_attrs}
        # Only generate 0/1 labels, to avoid generating labels bigger than the number of classes, which would lead to
        # an index out of range error when looking up the embedding of the class in the categorical feature tokenizer
        tab_attrs.update({attr: torch.randint(2, (2,)) for attr in self.tabular_cat_attrs})
        time_series_attrs = {
            (view, attr): torch.randn(2, self.hparams["data_params"].in_shape[CardinalTag.time_series_attrs][1])
            for view, attr in itertools.product(self.hparams["views"], self.hparams["time_series_attrs"])
        }
        time_series_notna_mask = torch.ones(
            (2, len(self.hparams["views"]) * len(self.hparams["time_series_attrs"])), dtype=torch.bool
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

                if self.hparams["ordinal_mode"] and target_tab_attr in TabularAttribute.ordinal_attrs():
                    # For ordinal targets, use a separate prediction head config
                    prediction_heads[target_tab_attr] = hydra.utils.instantiate(
                        self.hparams["model"]["ordinal_head"], num_logits=output_size
                    )
                else:
                    prediction_heads[target_tab_attr] = hydra.utils.instantiate(
                        self.hparams["model"]["prediction_head"], out_features=output_size
                    )

        return encoder, contrastive_head, prediction_heads

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizer to ignore parameters that should remain frozen (e.g. tokenizers)."""
        return super().configure_optimizers()

    @auto_move_data
    def tokenize(
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
        tokens_list, notna_mask_list = [], []

        # Tokenize the attributes
        if time_series_attrs:
            assert self.time_series_tokenizer is not None, "Time-series tokenizer must be configured to tokenize time-series."
            time_series_attrs_tokens = self.time_series_tokenizer(time_series_attrs)  # S * (N, ?) -> (N, S_ts, E)
            tokens_list.append(time_series_attrs_tokens)

            # Indicate that, when time-series tokens are requested, they are always available
            if time_series_notna_mask is None:
                time_series_notna_mask = torch.full(
                    time_series_attrs_tokens.shape[:2], True, device=time_series_attrs_tokens.device
                )

            # time_series_notna_mask = torch.full(
            #     time_series_attrs_tokens.shape[:2], True, device=time_series_attrs_tokens.device
            # )
            notna_mask_list.append(time_series_notna_mask)

        if tabular_attrs:
            assert self.tabular_tokenizer is not None, "Tabular tokenizer must be configured to tokenize tabular data."
            tab_attrs_tokens, tab_notna_mask = self.tabular_tokenizer(
                tabular_attrs=tabular_attrs,
                tabular_num_attrs=self.tabular_num_attrs,
                tabular_cat_attrs=self.tabular_cat_attrs,
            )

            tokens_list.append(tab_attrs_tokens)
            notna_mask_list.extend(tab_notna_mask)

        # Cast to float to make sure tokens are not represented using double
        tokens = torch.cat(tokens_list, dim=1).float()  # (N, S_ts + S_tab, E)
        # Cast to bool to make sure attention mask is represented by bool
        notna_mask = torch.cat(notna_mask_list, dim=1).bool()  # (N, S_ts + S_tab)

        return tokens, notna_mask

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

        if isinstance(self.mask_token, ParameterDict):
            mask_token = torch.stack(list(self.mask_token.values()))
        else:
            assert isinstance(self.mask_token, Parameter), "Mask token must be a Parameter or ParameterDict."
            mask_token = cast(Tensor, self.mask_token)

        if self.mask_token is not None:
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
        tokens: Tensor,
        avail_mask: Tensor,
        enable_augments: bool = False,
        output_intermediate: bool = False,
        output_unique: bool = False,
        output_all: bool = False,
    ) -> Tensor:
        """Embeds input sequences using the encoder model, optionally selecting/pooling output tokens for the embedding.

        Args:
            tokens: (N, S, E), Tokens to feed to the encoder.
            avail_mask: (N, S), Boolean mask indicating available (i.e. non-missing) tokens. Missing tokens can thus be
                treated distinctly from others (e.g. replaced w/ a specific mask).
            enable_augments: Whether to perform augments on the tokens (e.g. masking) to obtain a "corrupted" view for
                contrastive learning. Augments are already configured differently for training/testing (to avoid
                stochastic test-time predictions), so this parameter is simply useful to easily toggle augments on/off
                to obtain contrasting views.

        Returns: (N, E), Embeddings of the input sequences.
        """
        tokens = self.preprocess_tokens(tokens, avail_mask, enable_augments=enable_augments)

        if self.hparams["cls_token"]:
            # Add the CLS token to the end of each item in the batch
            tokens = self.cls_token(tokens)

        # Add positional encoding to the tokens
        tokens = self.positional_encoding(tokens)

        if self.separate_modality:
            # Split the sequence of tokens into tabular and time-series tokens
            ts_tokens, tab_tokens = tokens[:, : self.n_time_series_attrs], tokens[:, self.n_time_series_attrs :]

            if output_intermediate:
                return self.encoder(tab_tokens, ts_tokens, output_intermediate=output_intermediate)

            if self.hparams["cls_token"] and self.hparams["ts_cls_token"]:
                # Add the CLS token to the end of each item in the batch
                ts_tokens = self.cls_ts_token(ts_tokens)

                # Forward pass through the transformer encoder (starting with the cross-attention module)
                out_tokens, tab_cls_token, ts_cls_token = self.encoder(
                    tab_tokens, ts_tokens
                )  # Re-invert the order, as it is inverted in the Transformer forward pass

            else:
                # Forward pass through the transformer encoder (starting with the cross-attention module)
                out_tokens = self.encoder(tab_tokens, ts_tokens)

        else:
            if output_intermediate:
                return self.encoder(tokens, output_intermediate=output_intermediate)

            # Forward pass through the transformer encoder
            out_tokens = self.encoder(tokens)

        if output_unique:
            return out_tokens[:, : self.n_tabular_attrs, :]

        if output_all:
            return out_tokens

        if self.hparams["cls_token"]:
            # Only keep the CLS token (i.e. the last token) from the tokens outputted by the encoder
            out_features = out_tokens[:, -1, :]  # (N, S, E) -> (N, E)
            # if self.hparams["ts_cls_token"]:
            #     len_ts = self.n_time_series_attrs + 1
            #     out_ts_features = out_tokens[:, len_ts-1, :]  # (N, S, E) -> (N, E)
        elif self.hparams["sequence_pooling"]:
            # Perform sequence pooling of the transformers' output tokens
            out_features = self.sequence_pooling(out_tokens)  # (N, S, E) -> (N, E)
        else:
            # Return the output tokens as-is
            out_features = out_tokens.squeeze(1)  # (N, 1, E) -> (N, E)

        return out_features  # (N, E)

    @auto_move_data
    def forward(
        self,
        tabular_attrs: Dict[TabularAttribute, Tensor],
        time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor],
        time_series_notna_mask: Optional[Tensor],
        task: Literal["encode", "predict", "continuum_param", "continuum_tau"] = "encode",
        output_all: bool = False,
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
                ? * (M), Parameter of the unimodal logits distribution for ordinal targets.
            if `task` == 'continuum_tau`:
                ? * (M), Temperature used to control the sharpness of the unimodal logits distribution for ordinal
                         targets.
            if `task` == 'predict' (and the model includes prediction heads):
                ? * (N), Prediction for each target in `losses`.
        """
        if task != "encode" and not self.prediction_heads:
            raise ValueError(
                "You requested to perform a prediction task, but the model does not include any prediction heads."
            )
        if task in ["continuum_param", "continuum_tau"] and not self.hparams["ordinal_mode"]:
            raise ValueError(
                "You requested to obtain some parameters for ordinal attributes, but the model is not configured to "
                "predict ordinal targets. Either set `ordinal_mode` to `True` or change the requested inference task."
            )

        in_tokens, avail_mask = self.tokenize(
            tabular_attrs, time_series_attrs, time_series_notna_mask
        )  # (N, S, E), (N, S)
        # if self.contrastive_loss and self.hparams["contrastive_loss_weight"]:
        #     enable_proj = True
        out_features = self.encode(in_tokens, avail_mask, output_all=output_all)  # (N, S, E) -> (N, E)
        if output_all:
            return out_features

        # Early return if requested task requires no prediction heads
        if task == "encode":
            return out_features

        assert self.prediction_heads is not None, (
            "You requested to perform a prediction task, but the model does not include any prediction heads."
        )
        # Forward pass through each target's prediction head
        predictions = {attr: prediction_head(out_features) for attr, prediction_head in self.prediction_heads.items()}

        # Based on the requested task, extract and format the appropriate output of the prediction heads
        match task:
            case "predict":
                if self.hparams["ordinal_mode"]:
                    predictions = {
                        attr: pred[0] if attr in TabularAttribute.ordinal_attrs() else pred
                        for attr, pred in predictions.items()
                    }
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
        in_tokens, avail_mask = self.tokenize(tabular_attrs, time_series_attrs, None)  # (N, S, E), (N, S)
        return self.encode(in_tokens, avail_mask, output_intermediate=True)

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

        in_tokens, avail_mask = self.tokenize(
            tabular_attrs, time_series_attrs, time_series_notna_mask
        )  # (N, S, E), (N, S)

        metrics = {}
        losses = []
        if self.contrastive_loss is not None:  # run self-supervised contrastive step
            metrics.update(self._contrastive_shared_step(batch, batch_idx, in_tokens, avail_mask))
            losses.append(self.hparams["contrastive_loss_weight"] * metrics["cont_loss"])
            if self.orthogonal_loss:
                losses.append(self.hparams["orthogonal_loss_weight"] * metrics["orth_loss"])
        if self.predict_losses is not None:  # run fully-supervised prediction step
            metrics.update(self._prediction_shared_step(batch, batch_idx, in_tokens, avail_mask))
            losses.append(metrics["s_loss"])
        if self.reconstruction_loss_unique is not None:  # run reconstruction step
            metrics.update(self._reconstruction_step(batch, batch_idx, in_tokens, avail_mask))
            losses.append(self.hparams["reconstruction_loss_weight"] * metrics["rec_loss"])
        if self.inter_sample_loss is not None:  # run inter-sample step
            metrics.update(self._inter_sample_step(batch, batch_idx, in_tokens, avail_mask))
            losses.append(self.hparams["inter_sample_loss_weight"] * metrics["inter_loss"])
        if self.attribute_reg_loss is not None:  # run attribute regularization step
            metrics.update(self._attribute_reg_step(batch, batch_idx, in_tokens, avail_mask, tabular_attrs)) #type: ignore[arg-type]
            losses.append(self.hparams["attribute_reg_loss_weight"] * metrics["attr_reg_loss"])

        # Compute the sum of the (weighted) losses
        metrics["loss"] = sum(losses)
        return metrics

    def _prediction_shared_step(
        self,
        batch: PatientDataTarget,
        batch_idx: int,
        in_tokens: Tensor,
        avail_mask: Tensor,
    ) -> Dict[str, Tensor]:
        # Forward pass through the encoder without gradient computation to fine-tune only the prediction heads
        assert self.prediction_heads is not None, (
            "You requested to perform a prediction task, but the model does not include any prediction heads."
        )
        out_features = self.encode(in_tokens, avail_mask)
        predictions = {}
        for attr, prediction_head in self.prediction_heads.items():
            pred = prediction_head(out_features)
            if self.hparams["ordinal_mode"] and attr in TabularAttribute.ordinal_attrs():
                # For ordinal targets, extract the logits from the multiple outputs of classification head
                pred = pred[0]
            predictions[attr] = pred.squeeze(dim=1)
        # Print patient id related to predictions
        # print(f'Patient ID: {batch["id"]}, Predictions: {predictions}')
        # Compute the loss/metrics for each target attribute, ignoring items for which targets are missing
        losses, metrics = {}, {}
        for attr, loss in self.predict_losses.items():
            target, y_hat = batch[attr], predictions[attr]

            if attr in TabularAttribute.categorical_attrs():
                notna_mask = target != MISSING_CAT_ATTR
            else:  # attr in TabularAttribute.numerical_attrs():
                notna_mask = ~target.isnan()

            losses[f"{loss.__class__.__name__.lower().replace('loss', '')}/{attr}"] = loss(
                y_hat[notna_mask],
                # For BCE losses (e.g. `BCELoss`, BCEWithLogitsLoss`, etc.), the targets have to be floats,
                # so convert them from long to float
                target[notna_mask] if attr not in TabularAttribute.binary_attrs() else target[notna_mask].float(),
            )

            for metric_tag, metric in self.metrics[attr].items():
                # metric_res = metric(y_hat, target)

                # # For multiclass categorical attributes, metrics are not averaged by default, so log them for each
                # # class separately and then average them manually
                # if attr in TabularAttribute.categorical_attrs():
                #     # for class_label, metric_res_for_class in zip(TABULAR_CAT_ATTR_LABELS[attr], metric_res):
                #     #     metrics[f"{metric_tag}/{attr}/{class_label}"] = metric_res_for_class
                #     metric_res = metric_res.mean()

                # metrics[f"{metric_tag}/{attr}"] = metric_res
                metric.update(y_hat, target)

            # for metric_tag, metric in self.metrics[attr].items():
            #     metrics[f"{metric_tag}/{attr}"] = metric(y_hat[notna_mask], target[notna_mask])
            # for metric_tag, metric in self.metrics[attr].items():
            #     metric_res = metric(y_hat[notna_mask], target[notna_mask])

            #     # For multiclass categorical attributes, metrics are not averaged by default, so log them for each
            #     # class separately and then average them manually
            #     if attr in TabularAttribute.categorical_attrs():
            #         # for class_label, metric_res_for_class in zip(TABULAR_CAT_ATTR_LABELS[attr], metric_res):
            #         #     metrics[f"{metric_tag}/{attr}/{class_label}"] = metric_res_for_class
            #         metric_res = metric_res.mean()

            #     metrics[f"{metric_tag}/{attr}"] = metric_res

        # Reduce loss across the multiple targets
        if self.hparams["dummy_mode"]:
            losses["s_loss"] = 0.0
        else:
            losses["s_loss"] = torch.stack(list(losses.values())).mean()
        metrics.update(losses)

        return metrics

    def _reconstruction_step(
        self,
        batch: PatientDataTarget,
        batch_idx: int,
        in_tokens: Tensor,
        avail_mask: Tensor,
    ) -> Dict[str, Tensor]:
        assert self.reconstruction_loss_unique is not None, (
            "You requested to perform a reconstruction step, but the model does not include a reconstruction loss."
        )
        # Forward pass through the encoder
        tab_features = self.encode(in_tokens, avail_mask, output_unique=True)

        # Extract tabular attributes from the batch to get labels for the reconstruction loss
        tabular_attrs = {attr: attr_data for attr, attr_data in batch.items() if attr in self.hparams["tabular_attrs"]}
        num_attrs, cat_attrs = None, None
        if self.tabular_num_attrs:
            # Group the numerical attributes from the `tabular_attrs` input in a single tensor
            num_attrs = torch.hstack(
                [tabular_attrs[attr].unsqueeze(1) for attr in self.tabular_num_attrs]
            )  # (N, S_num)
        if self.tabular_cat_attrs:
            # Group the categorical attributes from the `tabular_attrs` input in a single tensor
            cat_attrs = torch.hstack(
                [tabular_attrs[attr].unsqueeze(1) for attr in self.tabular_cat_attrs]
            )  # (N, S_cat)

        #TODO: do the case where there is no categorical attributes nor numerical attributes
        assert (num_attrs is not None and cat_attrs is not None), "Tabular attributes must be provided for reconstruction step."
        tab_labels = torch.cat([torch.nan_to_num(num_attrs), cat_attrs.clip(0)], dim=1)  # (N, S_tab)

        # Compute the reconstruction loss
        metrics = {"rec_loss": self.reconstruction_loss_unique(tab_features, tab_labels)}

        return metrics

    def _inter_sample_step(
        self,
        batch: PatientDataTarget,
        batch_idx: int,
        in_tokens: Tensor,
        avail_mask: Tensor,
    ) -> Dict[str, Tensor]:
        assert self.inter_sample_loss is not None, (
            "You requested to perform an inter-sample step, but the model does not include an inter-sample loss."
        )
        metrics = {}

        # Forward pass through the encoder
        ts_avg, tab_unique_avg, tab_shared_avg = self.encode(in_tokens, avail_mask, output_intermediate=True)

        # Iterate on the attributes to get labels
        for attr, _ in self.predict_losses.items():
            target = batch[attr]

            if attr in TabularAttribute.categorical_attrs():
                notna_mask = target != MISSING_CAT_ATTR
            else:  # attr in TabularAttribute.numerical_attrs():
                notna_mask = ~target.isnan()

        # Compute the inter-sample loss/metrics
        metrics.update(
            {"inter_loss": self.inter_sample_loss(tab_unique_avg, ts_avg, target[notna_mask], tab_shared_avg)}
        )

        return metrics

    def _contrastive_shared_step(
        self,
        batch: PatientDataTarget,
        batch_idx: int,
        in_tokens: Tensor,
        avail_mask: Tensor,
    ) -> Dict[str, Tensor]:
        assert self.contrastive_loss is not None, (
            "You requested to perform a contrastive step, but the model does not include a contrastive loss."
        )
        # Extract features from the original view + from a view corrupted by augmentations
        # anchor_out_features = out_features
        # pre_ts_features, pre_features =  in_tokens[:, self.n_time_series_attrs, :], in_tokens[:, -1, :]
        # out_features, out_ts_features =  self.encode(in_tokens, avail_mask, enable_augments=False, alignment=True)
        # corrupted_out_features = self.encode(in_tokens, avail_mask, enable_augments=True)
        # ts_tokens, tab_tokens = in_tokens[:, : self.n_time_series_attrs], in_tokens[:, self.n_time_series_attrs :]

        # Get the average of the tokens for each modality

        ts_avg, tab_unique_avg, tab_shared_avg = self.encode(in_tokens, avail_mask, output_intermediate=True)

        # ts_tokens = self.time_series_lin_proj(ts_tokens)
        # ts_avg = torch.mean(ts_tokens, dim=1) # (N, E)
        # tab_tokens = self.tabular_lin_proj(tab_tokens)
        # tab_avg = torch.mean(tab_tokens, dim=1, keepdim=True).reshape(-1, 2, self.hparams["embed_dim"]) # (N, 2, E)

        # Iterate on the attributes to get labels (there is one attribute, hence the loop is only once)
        for attr, _ in self.predict_losses.items():
            target = batch[attr]

            if attr in TabularAttribute.categorical_attrs():
                notna_mask = target != MISSING_CAT_ATTR
            else:  # attr in TabularAttribute.numerical_attrs():
                notna_mask = ~target.isnan()
        # Compute the contrastive loss/metrics
        metrics = {
            "cont_loss": self.contrastive_loss(
                self.contrastive_head(tab_unique_avg),
                self.contrastive_head(tab_shared_avg),
                self.contrastive_head(ts_avg),
                target[notna_mask],
            )
        }
        if self.orthogonal_loss:
            metrics.update(
                {
                    "orth_loss": self.orthogonal_loss(
                        self.contrastive_head(tab_unique_avg), self.contrastive_head(tab_shared_avg)
                    )
                }
            )

        return metrics

    def _attribute_reg_step(
        self,
        batch: PatientDataInference,
        batch_idx: int,
        in_tokens: Tensor,
        avail_mask: Tensor,
        tabular_attrs: Dict[TabularAttribute, Tensor],
    ):
        """Computes the attribute regularization loss, if configured."""
        assert self.attribute_reg_loss is not None, (
            "You requested to perform an attribute regularization step, but the model does not include an attribute regularization loss."
        )
        metrics = {}
        # Forward pass through the encoder
        out_features = self.encode(in_tokens, avail_mask)  # Get the CLS token

        # Compute the attribute regularization loss
        metrics.update({"attr_reg_loss": self.attribute_reg_loss(out_features, tabular_attrs)})

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
    def predict_step( # noqa: D102
        self, batch: PatientData, batch_idx: int, dataloader_idx: int = 0
    ) -> Tuple[
        Tensor,
        Optional[Dict[TabularAttribute, Tensor]],
        Optional[Dict[TabularAttribute, Tensor]],
        Optional[Dict[TabularAttribute, Tensor]],
        Optional[Dict[str, Tensor]],
        Optional[Dict[str, Tensor]],
        ]:
        # Extract tabular and time-series attributes from the patient and add batch dimension
        # tabular_attrs = {attr: attr_data.unsqueeze(dim=0) for attr, attr_data in batch.items() if attr in self.hparams["tabular_attrs"]}
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
        # Add batch dimension to the time-series attributes
        # time_series_attrs = {
        #     (view, attr): attr_data.unsqueeze(dim=0)
        #     for (view, attr), attr_data in time_series_attrs.items()
        # }
        # time_series_attrs = {(view, attr): attr_data for (view, attr), attr_data in time_series_attrs.items()}
        # # Add batch dimension to the time-series notna mask
        # if time_series_notna_mask is not None and time_series_notna_mask.ndim == 1:
        #     time_series_notna_mask = time_series_notna_mask.unsqueeze(dim=0)

        # print(f"time series attrs: {time_series_attrs}")
        # print(f"batch id: {batch['id']}")
        # print(f"time series notna mask: {time_series_notna_mask}")

        # Encoder's output
        out_features = self(tabular_attrs, time_series_attrs, time_series_notna_mask)

        # If the model has targets to predict, output the predictions
        predictions = None
        if self.prediction_heads:
            predictions = self(tabular_attrs, time_series_attrs, time_series_notna_mask, task="predict")

        # If the model enforces constraint on ordinal targets, output the continuum parametrization
        continuum_params, continuum_taus = None, None
        if self.hparams["ordinal_mode"]:
            continuum_params = self(tabular_attrs, time_series_attrs, task="continuum_param")
            continuum_taus = self(tabular_attrs, time_series_attrs, task="continuum_tau")

        # Remove unnecessary batch dimension from the different outputs
        # (only do this once all downstream inferences have been performed)
        out_features = out_features.squeeze(dim=0)
        if predictions is not None:
            # Associate each prediction to the corresponding batch idx
            # predictions: (batch_size, ...)
            # batch["id"]: (batch_size,)

            # If predictions is a tensor, convert to list if needed
            predictions = {attr: prediction.squeeze(dim=0) for attr, prediction in predictions.items()}
            # pred_list = list(predictions.values())[0]
            # Associate each prediction to its id
            # pred_id_map = {id_: pred for id_, pred in zip(batch["id"], pred_list)}

        if self.hparams["ordinal_mode"]:
            assert continuum_params is not None, (
                "You requested to obtain some parameters for ordinal attributes, but the model is not configured to "
                "predict ordinal targets. Either set `ordinal_mode` to `True` or change the requested inference task."
            )
            assert continuum_taus is not None, (
                "You requested to obtain some parameters for ordinal attributes, but the model is not configured to "
                "predict ordinal targets. Either set `ordinal_mode` to `True` or change the requested inference task."
            )
            continuum_params = {
                attr: continuum_param.squeeze(dim=0) for attr, continuum_param in continuum_params.items()
            }
            continuum_taus = {attr: continuum_tau.squeeze(dim=0) for attr, continuum_tau in continuum_taus.items()}

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
            all_tokens = self(tabular_attrs, time_series_attrs, output_all=True)
            vectors_dict = {f"tab_shared_{i}": all_tokens[:, i, :] for i in range(13)}
            vectors_dict.update({f"ts_{i}": all_tokens[:, i + 13, :] for i in range(14)})
            vectors_dict.update({f"tab_specific_{i}": all_tokens[:, i + 13 + 14, :] for i in range(13)})
        else:
            vectors_dict = None
        # print(f"Patient ID: {batch['id']}, Predictions: {predictions}, ")
        return out_features, predictions, continuum_params, continuum_taus, latent_dict, vectors_dict
