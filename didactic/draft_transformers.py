import enum
import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, cast
import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import get_nn_module, MultiheadAttention, MultiheadCrossAttention
from .baselines import BidirectionalMultimodalAttention
from didactic.models.layers import CLSToken, PositionalEncoding, SequencePooling
from didactic.models.decoupling import DecouplingModule

ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = "Internal error. Please, open an issue."

def aggregate_tokens(
    ts_tokens: Tensor,
    tab_unique_tokens: Tensor,
    tab_shared_tokens: Tensor,
    mode: str = "average"
) -> Tuple[Tensor, Tensor, Tensor]:
    """Aggregate tokens from tabular and time-series attributes.
    
    Args:
        tab_unique_tokens: (N, U) Unique tokens from tabular attributes.
        tab_shared_tokens: (N, S) Shared tokens from tabular attributes.
        ts_tokens: (N, L) Tokens from time-series attributes.
        mode: Aggregation mode to use. One of: ['average', 'avg_token'].

    Returns:
        (N, E), (U, E), (S, E) Aggregated tokens from time-series, unique tabular, and shared tabular attributes.
    """

    if mode == "average":
        return ts_tokens.mean(dim=1), tab_unique_tokens.mean(dim=0), tab_shared_tokens.mean(dim=0)
    elif mode == "weighted_pooling":
        ts_sequence_pooling = SequencePooling(d_model=ts_tokens.shape[-1]).to(ts_tokens.device)
        tab_unique_pooling = SequencePooling(d_model=tab_unique_tokens.shape[-1]).to(tab_unique_tokens.device)
        tab_shared_pooling = SequencePooling(d_model=tab_shared_tokens.shape[-1]).to(tab_shared_tokens.device)
        return ts_sequence_pooling(ts_tokens), tab_unique_pooling(tab_unique_tokens), tab_shared_pooling(tab_shared_tokens)
    else:
        raise ValueError(f"Unexpected value for 'mode': {mode}. Use one of: ['average', 'weighted_pooling'].")

class FT_Transformer(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks
        
        layers = []

        
        if self.n_self_blocks:
            layers += [
                self._init_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks)
            ]

        if self.n_cross_blocks:
            layers += [
                self._init_cross_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks, self.n_self_blocks + self.n_cross_blocks)
            ]
        elif self.n_bidirectional_blocks:
            layers += [
                self._init_bidirectional_block(layer_idx)
                for layer_idx in range(self.n_self_blocks + self.n_cross_blocks, self.n_self_blocks + self.n_cross_blocks + self.n_bidirectional_blocks)
            ]

        self.blocks = nn.ModuleList(layers)

    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        if self.n_cross_blocks or self.n_bidirectional_blocks:
            layer = nn.ModuleDict()
            for modality_side in ["l", "r"]:
                layer.update(
                    {
                        f"{modality_side}_attention": MultiheadAttention(
                            d_token=self.d_token,
                            n_heads=self.attention_n_heads,
                            dropout=self.attention_dropout,
                            bias=True,
                            initialization=self.attention_initialization,
                        ),
                        f"{modality_side}_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                        f"{modality_side}_ffn": self.FFN(
                            d_token=self.d_token,
                            d_hidden=self.ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=self.ffn_dropout,
                            activation=self.ffn_activation,
                        ),
                        f"{modality_side}_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                    }
                )
                if layer_idx or not self.prenormalization or self.first_prenormalization:
                    layer[f"{modality_side}_attention_normalization"] = get_nn_module(self.attention_normalization)
                layer[f"{modality_side}_ffn_normalization"] = get_nn_module(self.ffn_normalization)
        else:
            layer = nn.ModuleDict(
                {
                    f"attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"ffn": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
            }
            # {   
            #     "l_cross_attention": MultiheadAttention(
            #         d_token=self.d_token,
            #         n_heads=self.attention_n_heads,
            #         dropout=self.attention_dropout,
            #         bias=True,
            #         initialization=self.attention_initialization,
            #     ),
            #     "r_cross_attention": MultiheadAttention(
            #         d_token=self.d_token,
            #         n_heads=self.attention_n_heads,
            #         dropout=self.attention_dropout,
            #         bias=True,
            #         initialization=self.attention_initialization,
            #     ),
            # }
        )

        for modality_side in ["l", "r"]:
            layer.update(
                {
                    f"{modality_side}_cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_0": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_0_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_cross_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_0_normalization"] = get_nn_module(self.ffn_normalization)

            layer.update(
                {
                    f"{modality_side}_self_attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"{modality_side}_self_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_1": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_1_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_self_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_1_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def _init_bidirectional_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "bidirectional_attention": BidirectionalMultimodalAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
            }
        )

        for modality_side in ["l", "r"]:
            layer.update(
                {
                    f"{modality_side}_bidirectional_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_bidirectional": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_bidirectional_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_bidirectional_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_bidirectional_normalization"] = get_nn_module(self.ffn_normalization)

        return layer


    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """

        if (self.n_cross_blocks or self.n_bidirectional_blocks) and x_context is None:
            raise ValueError(
                "x_context from which K and V are extracted, must be "
                "provided since the model includes cross-attention blocks."
            )
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        self_attention_blocks = self.blocks[: self.n_self_blocks]
        if self.n_cross_blocks:
            cross_attention_blocks = self.blocks[self.n_self_blocks :]
            bidirectional_attention_blocks = []
        elif self.n_bidirectional_blocks:
            cross_attention_blocks = []
            bidirectional_attention_blocks = self.blocks[self.n_self_blocks :]
        else:
            cross_attention_blocks = []
            bidirectional_attention_blocks = []

        for block in self_attention_blocks:
            block = cast(nn.ModuleDict, block)
            if x_context is not None:

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "l_attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["l_attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "l_attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "l_ffn", x, stage="self_attention")
                x_residual = block["l_ffn"](x_residual)
                x = self._end_residual(block, "l_ffn", x, x_residual, stage="self_attention")

                # Do the same for the context modality
                x_context_residual = self._start_residual(block, "r_attention", x_context, stage="self_attention")
                x_context_residual, _ = block["r_attention"](x_context_residual, x_context_residual)
                x_context = self._end_residual(block, "r_attention", x_context, x_context_residual, stage="self_attention")
                x_context_residual = self._start_residual(block, "r_ffn", x_context, stage="self_attention")
                x_context_residual = block["r_ffn"](x_context_residual)
                x_context = self._end_residual(block, "r_ffn", x_context, x_context_residual, stage="self_attention")

            else:
                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

        
        for block in cross_attention_blocks:
            block = cast(nn.ModuleDict, block)

            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "l_cross_attention", x, stage="cross_attention")
            x_context_residual = self._start_residual(block, "r_cross_attention", x_context, stage="cross_attention")

            # Forward pass through the cross-attention block
            x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "l_cross_attention", x, x_residual, stage="cross_attention")
            x_context = self._end_residual(block, "r_cross_attention", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_0", x, stage="cross_attention")
            x_residual = block["l_ffn_0"](x_residual)
            x = self._end_residual(block, "l_ffn_0", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_ffn_0", x_context, stage="cross_attention")
            x_context_residual = block["r_ffn_0"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_0", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the self-attention block inside the cross-attention module for both modalities
            x_residual = self._start_residual(block, "l_self_attention", x, stage="cross_attention")
            x_residual, _ = block["l_self_attention"](x_residual, x_residual)
            x = self._end_residual(block, "l_self_attention", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_self_attention", x_context, stage="cross_attention")
            x_context_residual, _ = block["r_self_attention"](x_context_residual, x_context_residual)
            x_context = self._end_residual(block, "r_self_attention", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_1", x, stage="cross_attention")
            x_residual = block["l_ffn_1"](x_residual)
            x = self._end_residual(block, "l_ffn_1", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_ffn_1", x_context, stage="cross_attention")
            x_context_residual = block["r_ffn_1"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_1", x_context, x_context_residual, stage="cross_attention")
        
        for block in bidirectional_attention_blocks:
            block = cast(nn.ModuleDict, block)

            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "l_bidirectional_attention", x, stage="bidirectional_attention")
            x_context_residual = self._start_residual(block, "r_bidirectional_attention", x_context, stage="bidirectional_attention")

            # Forward pass through the bidirectional attention block
            x_residual, x_context_residual = block["bidirectional_attention"](x_residual, x_context_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "l_bidirectional_attention", x, x_residual, stage="bidirectional_attention")
            x_context = self._end_residual(block, "r_bidirectional_attention", x_context, x_context_residual, stage="bidirectional_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_bidirectional", x, stage="bidirectional_attention")
            x_residual = block["l_ffn_bidirectional"](x_residual)
            x = self._end_residual(block, "l_ffn_bidirectional", x, x_residual, stage="bidirectional_attention")

            x_context_residual = self._start_residual(block, "r_ffn_bidirectional", x_context, stage="bidirectional_attention")
            x_context_residual = block["r_ffn_bidirectional"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_bidirectional", x_context, x_context_residual, stage="bidirectional_attention") 

        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x

        return output_tensor

class FT_Transformer_2UniFTs(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
        tabular_unimodal_encoder: str,
        ts_unimodal_encoder: str,
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        self.tabular_unimodal_encoder = get_nn_module(tabular_unimodal_encoder)
        self.ts_unimodal_encoder = get_nn_module(ts_unimodal_encoder)
        
        layers = []

        
        if self.n_self_blocks:
            layers += [
                self._init_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks)
            ]

        if self.n_cross_blocks:
            layers += [
                self._init_cross_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks, self.n_self_blocks + self.n_cross_blocks)
            ]

        elif self.n_bidirectional_blocks:
            layers += [
                self._init_bidirectional_block(layer_idx)
                for layer_idx in range(self.n_self_blocks + self.n_cross_blocks, self.n_self_blocks + self.n_cross_blocks + self.n_bidirectional_blocks)
            ]

        self.blocks = nn.ModuleList(layers)

    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        if self.n_cross_blocks or self.n_bidirectional_blocks:
            layer = nn.ModuleDict()
            for modality_side in ["l", "r"]:
                layer.update(
                    {
                        f"{modality_side}_attention": MultiheadAttention(
                            d_token=self.d_token,
                            n_heads=self.attention_n_heads,
                            dropout=self.attention_dropout,
                            bias=True,
                            initialization=self.attention_initialization,
                        ),
                        f"{modality_side}_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                        f"{modality_side}_ffn": self.FFN(
                            d_token=self.d_token,
                            d_hidden=self.ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=self.ffn_dropout,
                            activation=self.ffn_activation,
                        ),
                        f"{modality_side}_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                    }
                )
                if layer_idx or not self.prenormalization or self.first_prenormalization:
                    layer[f"{modality_side}_attention_normalization"] = get_nn_module(self.attention_normalization)
                layer[f"{modality_side}_ffn_normalization"] = get_nn_module(self.ffn_normalization)
        else:
            layer = nn.ModuleDict(
                {
                    f"attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"ffn": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
            }
        )

        for modality_side in ["l", "r"]:
            layer.update(
                {
                    f"{modality_side}_cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_0": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_0_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_cross_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_0_normalization"] = get_nn_module(self.ffn_normalization)

            layer.update(
                {
                    f"{modality_side}_self_attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"{modality_side}_self_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_1": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_1_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_self_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_1_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def _init_bidirectional_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "bidirectional_attention": BidirectionalMultimodalAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
            }
        )

        for modality_side in ["l", "r"]:
            layer.update(
                {
                    f"{modality_side}_bidirectional_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_bidirectional": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_bidirectional_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_bidirectional_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_bidirectional_normalization"] = get_nn_module(self.ffn_normalization)

        return layer


    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """

        # if (self.n_cross_blocks or self.n_bidirectional_blocks) and x_context is None:
        #     raise ValueError(
        #         "x_context from which K and V are extracted, must be "
        #         "provided since the model includes cross-attention blocks."
        #     )
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        self_attention_blocks = self.blocks[: self.n_self_blocks]
        if self.n_cross_blocks:
            cross_attention_blocks = self.blocks[self.n_self_blocks :]
            bidirectional_attention_blocks = []
        elif self.n_bidirectional_blocks:
            cross_attention_blocks = []
            bidirectional_attention_blocks = self.blocks[self.n_self_blocks :]

        # Extract the CLS token
        cls_token = x[:, -1, :]

        # Unimodal encoding for both modalities
        ts_tokens = self.ts_unimodal_encoder(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_unimodal_encoder(x[:, self.n_time_series_attrs :-1])

        if not self.n_cross_blocks and not self.n_bidirectional_blocks:
            x = torch.cat([tab_tokens, ts_tokens, cls_token.unsqueeze(1)], dim=1)
        else:
            x = torch.cat([tab_tokens, cls_token.unsqueeze(1)], dim=1)
            x_context = ts_tokens

        for block in self_attention_blocks:
            block = cast(nn.ModuleDict, block)
            if x_context is not None:

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "l_attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["l_attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "l_attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "l_ffn", x, stage="self_attention")
                x_residual = block["l_ffn"](x_residual)
                x = self._end_residual(block, "l_ffn", x, x_residual, stage="self_attention")

                # Do the same for the context modality
                x_context_residual = self._start_residual(block, "r_attention", x_context, stage="self_attention")
                x_context_residual, _ = block["r_attention"](x_context_residual, x_context_residual)
                x_context = self._end_residual(block, "r_attention", x_context, x_context_residual, stage="self_attention")
                x_context_residual = self._start_residual(block, "r_ffn", x_context, stage="self_attention")
                x_context_residual = block["r_ffn"](x_context_residual)
                x_context = self._end_residual(block, "r_ffn", x_context, x_context_residual, stage="self_attention")

            else:
                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

        
        for block in cross_attention_blocks:
            block = cast(nn.ModuleDict, block)

            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "l_cross_attention", x, stage="cross_attention")
            x_context_residual = self._start_residual(block, "r_cross_attention", x_context, stage="cross_attention")

            # Forward pass through the cross-attention block
            x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "l_cross_attention", x, x_residual, stage="cross_attention")
            x_context = self._end_residual(block, "r_cross_attention", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_0", x, stage="cross_attention")
            x_residual = block["l_ffn_0"](x_residual)
            x = self._end_residual(block, "l_ffn_0", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_ffn_0", x_context, stage="cross_attention")
            x_context_residual = block["r_ffn_0"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_0", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the self-attention block inside the cross-attention module for both modalities
            x_residual = self._start_residual(block, "l_self_attention", x, stage="cross_attention")
            x_residual, _ = block["l_self_attention"](x_residual, x_residual)
            x = self._end_residual(block, "l_self_attention", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_self_attention", x_context, stage="cross_attention")
            x_context_residual, _ = block["r_self_attention"](x_context_residual, x_context_residual)
            x_context = self._end_residual(block, "r_self_attention", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_1", x, stage="cross_attention")
            x_residual = block["l_ffn_1"](x_residual)
            x = self._end_residual(block, "l_ffn_1", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_ffn_1", x_context, stage="cross_attention")
            x_context_residual = block["r_ffn_1"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_1", x_context, x_context_residual, stage="cross_attention")
        
        for block in bidirectional_attention_blocks:
            block = cast(nn.ModuleDict, block)

            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "l_bidirectional_attention", x, stage="bidirectional_attention")
            x_context_residual = self._start_residual(block, "r_bidirectional_attention", x_context, stage="bidirectional_attention")

            # Forward pass through the bidirectional attention block
            x_residual, x_context_residual = block["bidirectional_attention"](x_residual, x_context_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "l_bidirectional_attention", x, x_residual, stage="bidirectional_attention")
            x_context = self._end_residual(block, "r_bidirectional_attention", x_context, x_context_residual, stage="bidirectional_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_bidirectional", x, stage="bidirectional_attention")
            x_residual = block["l_ffn_bidirectional"](x_residual)
            x = self._end_residual(block, "l_ffn_bidirectional", x, x_residual, stage="bidirectional_attention")

            x_context_residual = self._start_residual(block, "r_ffn_bidirectional", x_context, stage="bidirectional_attention")
            x_context_residual = block["r_ffn_bidirectional"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_bidirectional", x_context, x_context_residual, stage="bidirectional_attention") 

        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x

        return output_tensor

class FT_Alignment(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks
        
        self.tabular_lin_proj = nn.Linear(d_token, 2*d_token)
        self.time_series_lin_proj = nn.Linear(d_token, d_token)

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        layers = []
        
        if self.n_self_blocks:
            layers += [
                self._init_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks)
            ]

        if self.n_cross_blocks:
            layers += [
                self._init_cross_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks, self.n_self_blocks + self.n_cross_blocks)
            ]

        self.blocks = nn.ModuleList(layers)

    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        if self.n_cross_blocks or self.n_bidirectional_blocks:
            layer = nn.ModuleDict()
            for modality_side in ["l", "r"]:
                layer.update(
                    {
                        f"{modality_side}_attention": MultiheadAttention(
                            d_token=self.d_token,
                            n_heads=self.attention_n_heads,
                            dropout=self.attention_dropout,
                            bias=True,
                            initialization=self.attention_initialization,
                        ),
                        f"{modality_side}_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                        f"{modality_side}_ffn": self.FFN(
                            d_token=self.d_token,
                            d_hidden=self.ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=self.ffn_dropout,
                            activation=self.ffn_activation,
                        ),
                        f"{modality_side}_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                    }
                )
                if layer_idx or not self.prenormalization or self.first_prenormalization:
                    layer[f"{modality_side}_attention_normalization"] = get_nn_module(self.attention_normalization)
                layer[f"{modality_side}_ffn_normalization"] = get_nn_module(self.ffn_normalization)
        else:
            layer = nn.ModuleDict(
                {
                    f"attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"ffn": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
            }
        )

        for modality_side in ["l", "r"]:
            layer.update(
                {
                    f"{modality_side}_cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_0": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_0_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_cross_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_0_normalization"] = get_nn_module(self.ffn_normalization)

            layer.update(
                {
                    f"{modality_side}_self_attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"{modality_side}_self_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_1": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_1_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_self_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_1_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None, output_intermediate: bool = False) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """

        if (self.n_cross_blocks or self.n_bidirectional_blocks) and x_context is None:
            raise ValueError(
                "x_context from which K and V are extracted, must be "
                "provided since the model includes cross-attention blocks."
            )
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        self_attention_blocks = self.blocks[: self.n_self_blocks]
        cross_attention_blocks = self.blocks[self.n_self_blocks :]
        bidirectional_attention_blocks = [] #FIXME: build bidirectional attention blocks later

        # Linear projections for alignment
        ts_tokens = self.time_series_lin_proj(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_lin_proj(x[:, self.n_time_series_attrs :-1])
        cls_tokens = x[:, -1, :]
        tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,::2,:]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,1::2,:]

        if output_intermediate:
            return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)

        # Set the unique tokens at the beginning of the sequence for extraction purposes
        x = torch.cat([tab_tokens_unique, tab_tokens_shared, ts_tokens, cls_tokens.unsqueeze(1)], dim=1)

        for block in self_attention_blocks:
            block = cast(nn.ModuleDict, block)
             
            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "attention", x, stage="self_attention")

            # Forward pass through the self-attention block
            x_residual, _ = block["attention"](x_residual, x_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
            x_residual = block["ffn"](x_residual)
            x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x

        return output_tensor

class FT_Alignment_2UniFTs(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
        tabular_unimodal_encoder: str,
        ts_unimodal_encoder: str,
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks
        
        self.tabular_lin_proj = nn.Linear(d_token, 2*d_token)
        self.time_series_lin_proj = nn.Linear(d_token, d_token)

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        self.tabular_unimodal_encoder = get_nn_module(tabular_unimodal_encoder)
        self.ts_unimodal_encoder = get_nn_module(ts_unimodal_encoder)

        layers = []
        
        if self.n_self_blocks:
            layers += [
                self._init_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks)
            ]

        if self.n_cross_blocks:
            layers += [
                self._init_cross_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks, self.n_self_blocks + self.n_cross_blocks)
            ]

        self.blocks = nn.ModuleList(layers)

    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        if self.n_cross_blocks or self.n_bidirectional_blocks:
            layer = nn.ModuleDict()
            for modality_side in ["l", "r"]:
                layer.update(
                    {
                        f"{modality_side}_attention": MultiheadAttention(
                            d_token=self.d_token,
                            n_heads=self.attention_n_heads,
                            dropout=self.attention_dropout,
                            bias=True,
                            initialization=self.attention_initialization,
                        ),
                        f"{modality_side}_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                        f"{modality_side}_ffn": self.FFN(
                            d_token=self.d_token,
                            d_hidden=self.ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=self.ffn_dropout,
                            activation=self.ffn_activation,
                        ),
                        f"{modality_side}_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                    }
                )
                if layer_idx or not self.prenormalization or self.first_prenormalization:
                    layer[f"{modality_side}_attention_normalization"] = get_nn_module(self.attention_normalization)
                layer[f"{modality_side}_ffn_normalization"] = get_nn_module(self.ffn_normalization)
        else:
            layer = nn.ModuleDict(
                {
                    f"attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"ffn": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
            }
        )

        for modality_side in ["l", "r"]:
            layer.update(
                {
                    f"{modality_side}_cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_0": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_0_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_cross_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_0_normalization"] = get_nn_module(self.ffn_normalization)

            layer.update(
                {
                    f"{modality_side}_self_attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"{modality_side}_self_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_1": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_1_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_self_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_1_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None, output_intermediate: bool = False) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """

        if (self.n_cross_blocks or self.n_bidirectional_blocks) and x_context is None:
            raise ValueError(
                "x_context from which K and V are extracted, must be "
                "provided since the model includes cross-attention blocks."
            )
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        self_attention_blocks = self.blocks[: self.n_self_blocks]
        cross_attention_blocks = self.blocks[self.n_self_blocks :]
        bidirectional_attention_blocks = [] #FIXME: build bidirectional attention blocks later

        # Extract the class tokens
        cls_tokens = x[:, -1, :]

        # Unimodal encoding for both modalities
        ts_tokens = self.ts_unimodal_encoder(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_unimodal_encoder(x[:, self.n_time_series_attrs :-1])

        # Linear projections for alignment
        ts_tokens = self.time_series_lin_proj(ts_tokens)
        tab_tokens = self.tabular_lin_proj(tab_tokens)
        
        tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,::2,:]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,1::2,:]
        # tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,:self.n_tabular_attrs,:]
        # tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,self.n_tabular_attrs:,:]

        if output_intermediate:
            return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)

        # Set the unique tokens at the beginning of the sequence for extraction purposes
        x = torch.cat([tab_tokens_unique, tab_tokens_shared, ts_tokens, cls_tokens.unsqueeze(1)], dim=1)

        for block in self_attention_blocks:
            block = cast(nn.ModuleDict, block)
            
            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "attention", x, stage="self_attention")

            # Forward pass through the self-attention block
            x_residual, _ = block["attention"](x_residual, x_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
            x_residual = block["ffn"](x_residual)
            x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x

        return output_tensor

class FT_Alignment_2UniFTs_CrossAtt(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
        tabular_unimodal_encoder: str,
        ts_unimodal_encoder: str,
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        assert n_cross_blocks, "Cross attention blocks must be present in the Alignemtn CrossAttention model"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks
        
        self.tabular_lin_proj = nn.Linear(d_token, 2*d_token)
        self.time_series_lin_proj = nn.Linear(d_token, d_token)

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        self.tabular_unimodal_encoder = get_nn_module(tabular_unimodal_encoder)
        self.ts_unimodal_encoder = get_nn_module(ts_unimodal_encoder)

        layers = []
        
        if self.n_self_blocks:
            layers += [
                self._init_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks)
            ]

        if self.n_cross_blocks:
            layers += [
                self._init_cross_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks, self.n_self_blocks + self.n_cross_blocks)
            ]

        self.blocks = nn.ModuleList(layers)

    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        if self.n_cross_blocks or self.n_bidirectional_blocks:
            layer = nn.ModuleDict()
            for modality_side in ["l", "r"]:
                layer.update(
                    {
                        f"{modality_side}_attention": MultiheadAttention(
                            d_token=self.d_token,
                            n_heads=self.attention_n_heads,
                            dropout=self.attention_dropout,
                            bias=True,
                            initialization=self.attention_initialization,
                        ),
                        f"{modality_side}_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                        f"{modality_side}_ffn": self.FFN(
                            d_token=self.d_token,
                            d_hidden=self.ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=self.ffn_dropout,
                            activation=self.ffn_activation,
                        ),
                        f"{modality_side}_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                    }
                )
                if layer_idx or not self.prenormalization or self.first_prenormalization:
                    layer[f"{modality_side}_attention_normalization"] = get_nn_module(self.attention_normalization)
                layer[f"{modality_side}_ffn_normalization"] = get_nn_module(self.ffn_normalization)
        else:
            layer = nn.ModuleDict(
                {
                    f"attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"ffn": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
            }
        )

        for modality_side in ["l", "r"]:
            layer.update(
                {
                    f"{modality_side}_cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_0": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_0_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_cross_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_0_normalization"] = get_nn_module(self.ffn_normalization)

            layer.update(
                {
                    f"{modality_side}_self_attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"{modality_side}_self_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_1": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_1_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_self_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_1_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None, output_intermediate: bool = False) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        self_attention_blocks = self.blocks[: self.n_self_blocks]
        cross_attention_blocks = self.blocks[self.n_self_blocks :]
        bidirectional_attention_blocks = [] #FIXME: build bidirectional attention blocks later

        # Extract the class tokens
        cls_tokens = x[:, -1, :]

        # Unimodal encoding for both modalities
        ts_tokens = self.ts_unimodal_encoder(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_unimodal_encoder(x[:, self.n_time_series_attrs :-1])

        # Linear projections for alignment
        ts_tokens = self.time_series_lin_proj(ts_tokens)
        tab_tokens = self.tabular_lin_proj(tab_tokens)
        
        tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,::2,:]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,1::2,:]

        if output_intermediate:
            return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)

        # x = torch.cat([ts_tokens, tab_tokens_unique, tab_tokens_shared, cls_tokens.unsqueeze(1)], dim=1)
        x = torch.cat([tab_tokens_shared, ts_tokens, cls_tokens.unsqueeze(1)], dim=1)
        x_context = tab_tokens_unique

        for block in self_attention_blocks:
            block = cast(nn.ModuleDict, block)

            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "l_attention", x, stage="self_attention")

            # Forward pass through the self-attention block
            x_residual, _ = block["l_attention"](x_residual, x_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "l_attention", x, x_residual, stage="self_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn", x, stage="self_attention")
            x_residual = block["l_ffn"](x_residual)
            x = self._end_residual(block, "l_ffn", x, x_residual, stage="self_attention")

            # Do the same for the context modality
            x_context_residual = self._start_residual(block, "r_attention", x_context, stage="self_attention")
            x_context_residual, _ = block["r_attention"](x_context_residual, x_context_residual)
            x_context = self._end_residual(block, "r_attention", x_context, x_context_residual, stage="self_attention")
            x_context_residual = self._start_residual(block, "r_ffn", x_context, stage="self_attention")
            x_context_residual = block["r_ffn"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn", x_context, x_context_residual, stage="self_attention")

        for block in cross_attention_blocks:
            block = cast(nn.ModuleDict, block)

            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "l_cross_attention", x, stage="cross_attention")
            x_context_residual = self._start_residual(block, "r_cross_attention", x_context, stage="cross_attention")

            # Forward pass through the cross-attention block
            x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "l_cross_attention", x, x_residual, stage="cross_attention")
            x_context = self._end_residual(block, "r_cross_attention", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_0", x, stage="cross_attention")
            x_residual = block["l_ffn_0"](x_residual)
            x = self._end_residual(block, "l_ffn_0", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_ffn_0", x_context, stage="cross_attention")
            x_context_residual = block["r_ffn_0"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_0", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the self-attention block inside the cross-attention module for both modalities
            x_residual = self._start_residual(block, "l_self_attention", x, stage="cross_attention")
            x_residual, _ = block["l_self_attention"](x_residual, x_residual)
            x = self._end_residual(block, "l_self_attention", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_self_attention", x_context, stage="cross_attention")
            x_context_residual, _ = block["r_self_attention"](x_context_residual, x_context_residual)
            x_context = self._end_residual(block, "r_self_attention", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_1", x, stage="cross_attention")
            x_residual = block["l_ffn_1"](x_residual)
            x = self._end_residual(block, "l_ffn_1", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_ffn_1", x_context, stage="cross_attention")
            x_context_residual = block["r_ffn_1"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_1", x_context, x_context_residual, stage="cross_attention")
        
        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x

        return output_tensor

class FT_Alignment_2UniFTs_BiDirectional(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
        tabular_unimodal_encoder: str,
        ts_unimodal_encoder: str,
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks
        
        self.tabular_lin_proj = nn.Linear(d_token, 2*d_token)
        self.time_series_lin_proj = nn.Linear(d_token, d_token)

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        self.tabular_unimodal_encoder = get_nn_module(tabular_unimodal_encoder)
        self.ts_unimodal_encoder = get_nn_module(ts_unimodal_encoder)

        layers = []
        
        if self.n_self_blocks:
            layers += [
                self._init_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks)
            ]

        if self.n_cross_blocks:
            layers += [
                self._init_cross_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks, self.n_self_blocks + self.n_cross_blocks)
            ]

        elif self.n_bidirectional_blocks:
            layers += [
                self._init_bidirectional_block(layer_idx)
                for layer_idx in range(self.n_self_blocks, self.n_self_blocks + self.n_bidirectional_blocks)
            ]

        self.blocks = nn.ModuleList(layers)

    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        if self.n_cross_blocks or self.n_bidirectional_blocks:
            layer = nn.ModuleDict()
            for modality_side in ["l", "r"]:
                layer.update(
                    {
                        f"{modality_side}_attention": MultiheadAttention(
                            d_token=self.d_token,
                            n_heads=self.attention_n_heads,
                            dropout=self.attention_dropout,
                            bias=True,
                            initialization=self.attention_initialization,
                        ),
                        f"{modality_side}_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                        f"{modality_side}_ffn": self.FFN(
                            d_token=self.d_token,
                            d_hidden=self.ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=self.ffn_dropout,
                            activation=self.ffn_activation,
                        ),
                        f"{modality_side}_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                    }
                )
                if layer_idx or not self.prenormalization or self.first_prenormalization:
                    layer[f"{modality_side}_attention_normalization"] = get_nn_module(self.attention_normalization)
                layer[f"{modality_side}_ffn_normalization"] = get_nn_module(self.ffn_normalization)
        else:
            layer = nn.ModuleDict(
                {
                    f"attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"ffn": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
            }
        )

        for modality_side in ["l", "r"]:
            layer.update(
                {
                    f"{modality_side}_cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_0": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_0_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_cross_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_0_normalization"] = get_nn_module(self.ffn_normalization)

            layer.update(
                {
                    f"{modality_side}_self_attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"{modality_side}_self_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_1": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_1_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_self_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_1_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def _init_bidirectional_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "bidirectional_attention": BidirectionalMultimodalAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
            }
        )

        for modality_side in ["l", "r"]:
            layer.update(
                {
                    f"{modality_side}_bidirectional_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_bidirectional": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_bidirectional_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_bidirectional_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_bidirectional_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None, output_intermediate: bool = False) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        self_attention_blocks = self.blocks[: self.n_self_blocks]
        bidirectional_attention_blocks = self.blocks[self.n_self_blocks :]

        # Extract the class tokens
        cls_tokens = x[:, -1, :]

        # Unimodal encoding for both modalities
        ts_tokens = self.ts_unimodal_encoder(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_unimodal_encoder(x[:, self.n_time_series_attrs :-1])

        # Linear projections for alignment
        ts_tokens = self.time_series_lin_proj(ts_tokens)
        tab_tokens = self.tabular_lin_proj(tab_tokens)
        
        tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,::2,:]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,1::2,:]

        if output_intermediate:
            return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)

        # x = torch.cat([ts_tokens, tab_tokens_unique, tab_tokens_shared, cls_tokens.unsqueeze(1)], dim=1)
        x = torch.cat([tab_tokens_unique, cls_tokens.unsqueeze(1)], dim=1)
        x_context = torch.cat([tab_tokens_shared, ts_tokens], dim=1)

        for block in self_attention_blocks:
            block = cast(nn.ModuleDict, block)

            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "l_attention", x, stage="self_attention")

            # Forward pass through the self-attention block
            x_residual, _ = block["l_attention"](x_residual, x_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "l_attention", x, x_residual, stage="self_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn", x, stage="self_attention")
            x_residual = block["l_ffn"](x_residual)
            x = self._end_residual(block, "l_ffn", x, x_residual, stage="self_attention")

            # Do the same for the context modality
            x_context_residual = self._start_residual(block, "r_attention", x_context, stage="self_attention")
            x_context_residual, _ = block["r_attention"](x_context_residual, x_context_residual)
            x_context = self._end_residual(block, "r_attention", x_context, x_context_residual, stage="self_attention")
            x_context_residual = self._start_residual(block, "r_ffn", x_context, stage="self_attention")
            x_context_residual = block["r_ffn"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn", x_context, x_context_residual, stage="self_attention")

        for block in bidirectional_attention_blocks:
            block = cast(nn.ModuleDict, block)

            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "l_bidirectional_attention", x, stage="bidirectional_attention")
            x_context_residual = self._start_residual(block, "r_bidirectional_attention", x_context, stage="bidirectional_attention")

            # Forward pass through the bidirectional attention block
            x_residual, x_context_residual = block["bidirectional_attention"](x_residual, x_context_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "l_bidirectional_attention", x, x_residual, stage="bidirectional_attention")
            x_context = self._end_residual(block, "r_bidirectional_attention", x_context, x_context_residual, stage="bidirectional_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_bidirectional", x, stage="bidirectional_attention")
            x_residual = block["l_ffn_bidirectional"](x_residual)
            x = self._end_residual(block, "l_ffn_bidirectional", x, x_residual, stage="bidirectional_attention")

            x_context_residual = self._start_residual(block, "r_ffn_bidirectional", x_context, stage="bidirectional_attention")
            x_context_residual = block["r_ffn_bidirectional"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_bidirectional", x_context, x_context_residual, stage="bidirectional_attention")
        
        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x

        return output_tensor

class FT_Interleaved(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        freeze_self_attention: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks
        
        layers = []
        total_blocks = max(self.n_self_blocks, self.n_cross_blocks)

        if self.n_cross_blocks > 1:
            for layer_idx in range(total_blocks):
                if layer_idx < self.n_self_blocks:
                    layers.append(self._init_attention_block(layer_idx))
                
                if layer_idx < self.n_cross_blocks:
                    layers.append(self._init_cross_attention_block(layer_idx))
        elif self.n_cross_blocks == 1:
            for layer_idx in range(self.n_self_blocks):
                layers.append(self._init_attention_block(layer_idx))
            self.unique_cross_block = self._init_cross_attention_block(0)

        self.blocks = nn.ModuleList(layers)

        if freeze_self_attention:
            self.freeze_self_attention()


    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                f"attention": MultiheadAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                f"ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
        layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                "cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                "cross_ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                "cross_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer["cross_attention_normalization"] = get_nn_module(self.attention_normalization)
        layer["cross_ffn_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def freeze_self_attention(self):
        for block in self.blocks:
            if "attention" in block:
                block["attention"].requires_grad_(False)
                if "attention_normalization" in block:
                    block["attention_normalization"].requires_grad_(False)

    # def _init_bidirectional_block(self, layer_idx: int) -> nn.ModuleDict:
    #     layer = nn.ModuleDict(
    #         {
    #             "bidirectional_attention": BidirectionalMultimodalAttention(
    #                 d_token=self.d_token,
    #                 n_heads=self.attention_n_heads,
    #                 dropout=self.attention_dropout,
    #                 bias=True,
    #                 initialization=self.attention_initialization,
    #             ),
    #         }
    #     )

    #     layer.update(
    #         {
    #             f"bidirectional_attention_residual_dropout": nn.Dropout(self.residual_dropout),
    #             f"bidirectional_ffn": self.FFN(
    #                 d_token=self.d_token,
    #                 d_hidden=self.ffn_d_hidden,
    #                 bias_first=True,
    #                 bias_second=True,
    #                 dropout=self.ffn_dropout,
    #                 activation=self.ffn_activation,
    #             ),
    #             f"bidirectional_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
    #         }
    #     )
    #     if layer_idx or not self.prenormalization or self.first_prenormalization:
    #         layer[f"bidirectional_attention_normalization"] = get_nn_module(self.attention_normalization)
    #     layer[f"bidirectional_ffn_normalization"] = get_nn_module(self.ffn_normalization)

    #     return layer


    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            # case "bidirectional_attention":
            #     assert layer_name in [
            #         "bidirectional_attention",
            #         "bidirectional_ffn",
            #     ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "bidirectional_attention",
                    "bidirectional_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """

        if (self.n_cross_blocks or self.n_bidirectional_blocks) and x_context is None:
            raise ValueError(
                "x_context from which K and V are extracted, must be "
                "provided since the model includes cross-attention blocks."
            )
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        for block in self.blocks:
            if "cross_attention" not in block:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

                if self.unique_cross_block:
                    block = self.unique_cross_block

                    block = cast(nn.ModuleDict, block)

                    # Normalize the tokens from both modalities if prenormalization is enabled
                    x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                    x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                    # Forward pass through the cross-attention block
                    x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                    # Residual connections after the attention layer for both modalities
                    x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                    x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                    # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                    x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                    x_residual = block["cross_ffn"](x_residual)
                    x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                    x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                    x_context_residual = block["cross_ffn"](x_context_residual)
                    x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention")

            else:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                # Forward pass through the cross-attention block
                x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                x_residual = block["cross_ffn"](x_residual)
                x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                x_context_residual = block["cross_ffn"](x_context_residual)
                x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention") 

        # Concatenation is only for hook purposes
        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x

        return output_tensor

class FT_Interleaved_Alignment(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        freeze_self_attention: bool = False,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks

        self.tabular_lin_proj = nn.Linear(d_token, 2*d_token)
        self.time_series_lin_proj = nn.Linear(d_token, d_token)

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        
        layers = []
        total_blocks = max(self.n_self_blocks, self.n_cross_blocks)

        if self.n_cross_blocks > 1:
            for layer_idx in range(total_blocks):
                if layer_idx < self.n_self_blocks:
                    layers.append(self._init_attention_block(layer_idx))
                
                if layer_idx < self.n_cross_blocks:
                    layers.append(self._init_cross_attention_block(layer_idx))
        elif self.n_cross_blocks == 1:
            for layer_idx in range(self.n_self_blocks):
                layers.append(self._init_attention_block(layer_idx))
            self.unique_cross_block = self._init_cross_attention_block(0)

        self.blocks = nn.ModuleList(layers)

        if freeze_self_attention:
            self.freeze_self_attention()


    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                f"attention": MultiheadAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                f"ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
        layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                "cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                "cross_ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                "cross_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer["cross_attention_normalization"] = get_nn_module(self.attention_normalization)
        layer["cross_ffn_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def freeze_self_attention(self):
        for block in self.blocks:
            if "attention" in block:
                block["attention"].requires_grad_(False)
                if "attention_normalization" in block:
                    block["attention_normalization"].requires_grad_(False)

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "bidirectional_attention",
                    "bidirectional_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None, output_intermediate: bool = False) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        # Linear projections for alignment
        ts_tokens = self.time_series_lin_proj(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_lin_proj(x[:, self.n_time_series_attrs :-1])
        cls_tokens = x[:, -1, :]
        tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,::2,:]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,1::2,:]

        if output_intermediate:
            return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)

        # x = torch.cat([ts_tokens, tab_tokens_unique, tab_tokens_shared, cls_tokens.unsqueeze(1)], dim=1)
        x = torch.cat([tab_tokens_shared, ts_tokens, cls_tokens.unsqueeze(1)], dim=1)
        x_context = tab_tokens_unique

        for layer_idx, block in enumerate(self.blocks):
            if "cross_attention" not in block:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

                if self.unique_cross_block and layer_idx < self.n_self_blocks-1:
                    block = self.unique_cross_block

                    block = cast(nn.ModuleDict, block)

                    # Normalize the tokens from both modalities if prenormalization is enabled
                    x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                    x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                    # Forward pass through the cross-attention block
                    x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                    # Residual connections after the attention layer for both modalities
                    x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                    x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                    # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                    x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                    x_residual = block["cross_ffn"](x_residual)
                    x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                    x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                    x_context_residual = block["cross_ffn"](x_context_residual)
                    x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention")

            else:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                # Forward pass through the cross-attention block
                x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                x_residual = block["cross_ffn"](x_residual)
                x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                x_context_residual = block["cross_ffn"](x_context_residual)
                x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention") 

        # Concatenation is only for hook purposes
        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x

        return output_tensor

class FT_Interleaved_2UniFTs(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        freeze_self_attention: bool = False,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
        tabular_unimodal_encoder: str,
        ts_unimodal_encoder: str,
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks

        self.tabular_lin_proj = nn.Linear(d_token, 2*d_token)
        self.time_series_lin_proj = nn.Linear(d_token, d_token)

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        self.tabular_unimodal_encoder = get_nn_module(tabular_unimodal_encoder)
        self.ts_unimodal_encoder = get_nn_module(ts_unimodal_encoder)

        
        layers = []
        total_blocks = max(self.n_self_blocks, self.n_cross_blocks)

        if self.n_cross_blocks > 1:
            for layer_idx in range(total_blocks):
                if layer_idx < self.n_self_blocks:
                    layers.append(self._init_attention_block(layer_idx))
                
                if layer_idx < self.n_cross_blocks:
                    layers.append(self._init_cross_attention_block(layer_idx))
        elif self.n_cross_blocks == 1:
            for layer_idx in range(self.n_self_blocks):
                layers.append(self._init_attention_block(layer_idx))
            self.unique_cross_block = self._init_cross_attention_block(0)

        self.blocks = nn.ModuleList(layers)

        if freeze_self_attention:
            self.freeze_self_attention()


    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                f"attention": MultiheadAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                f"ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
        layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                "cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                "cross_ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                "cross_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer["cross_attention_normalization"] = get_nn_module(self.attention_normalization)
        layer["cross_ffn_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def freeze_self_attention(self):
        for block in self.blocks:
            if "attention" in block:
                block["attention"].requires_grad_(False)
                if "attention_normalization" in block:
                    block["attention_normalization"].requires_grad_(False)

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "bidirectional_attention",
                    "bidirectional_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None, output_intermediate: bool = False) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        # Extract the class tokens
        cls_tokens = x[:, -1, :]

        # Unimodal encoding for both modalities
        ts_tokens = self.ts_unimodal_encoder(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_unimodal_encoder(x[:, self.n_time_series_attrs :-1])

        # Linear projections for alignment
        ts_tokens = self.time_series_lin_proj(ts_tokens)
        tab_tokens = self.tabular_lin_proj(tab_tokens)
        
        tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,::2,:]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,1::2,:]

        if output_intermediate:
            return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)

        # x = torch.cat([ts_tokens, tab_tokens_unique, tab_tokens_shared, cls_tokens.unsqueeze(1)], dim=1)
        x = torch.cat([tab_tokens_shared, ts_tokens, cls_tokens.unsqueeze(1)], dim=1)
        x_context = tab_tokens_unique

        for layer_idx, block in enumerate(self.blocks):
            if "cross_attention" not in block:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

                if self.unique_cross_block and layer_idx < self.n_self_blocks-1:
                    block = self.unique_cross_block

                    block = cast(nn.ModuleDict, block)

                    # Normalize the tokens from both modalities if prenormalization is enabled
                    x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                    x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                    # Forward pass through the cross-attention block
                    x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                    # Residual connections after the attention layer for both modalities
                    x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                    x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                    # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                    x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                    x_residual = block["cross_ffn"](x_residual)
                    x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                    x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                    x_context_residual = block["cross_ffn"](x_context_residual)
                    x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention")

            else:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                # Forward pass through the cross-attention block
                x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                x_residual = block["cross_ffn"](x_residual)
                x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                x_context_residual = block["cross_ffn"](x_context_residual)
                x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention") 

        # Concatenation is only for hook purposes
        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x

        return output_tensor

class FT_Interleaved_2UniFTs_Inverted(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        freeze_self_attention: bool = False,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
        tabular_unimodal_encoder: str,
        ts_unimodal_encoder: str,
        decoupling_method: str = "linear",
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        self.tabular_unimodal_encoder = get_nn_module(tabular_unimodal_encoder)
        self.ts_unimodal_encoder = get_nn_module(ts_unimodal_encoder)

        self.decoupling_module = DecouplingModule(
            ts_input_size=d_token,
            ts_proj_size=d_token,
            tab_input_size=d_token,
            tab_proj_size=2*d_token,
            decoupling_method=decoupling_method
        )
        
        layers = []
        total_blocks = max(self.n_self_blocks, self.n_cross_blocks)

        if self.n_cross_blocks > 1:
            for layer_idx in range(total_blocks):
                if layer_idx < self.n_self_blocks:
                    layers.append(self._init_attention_block(layer_idx))
                
                if layer_idx < self.n_cross_blocks:
                    layers.append(self._init_cross_attention_block(layer_idx))
        elif self.n_cross_blocks == 1:
            for layer_idx in range(self.n_self_blocks):
                layers.append(self._init_attention_block(layer_idx))
            self.unique_cross_block = self._init_cross_attention_block(0)

        self.blocks = nn.ModuleList(layers)

        if freeze_self_attention:
            self.freeze_self_attention()


    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                f"attention": MultiheadAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                f"ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
        layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                "cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                "cross_ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                "cross_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer["cross_attention_normalization"] = get_nn_module(self.attention_normalization)
        layer["cross_ffn_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def freeze_self_attention(self):
        for block in self.blocks:
            if "attention" in block:
                block["attention"].requires_grad_(False)
                if "attention_normalization" in block:
                    block["attention_normalization"].requires_grad_(False)

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "bidirectional_attention",
                    "bidirectional_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None, output_intermediate: bool = False) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        # Extract the class tokens
        cls_tokens = x[:, -1, :]

        # Unimodal encoding for both modalities
        ts_tokens = self.ts_unimodal_encoder(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_unimodal_encoder(x[:, self.n_time_series_attrs :-1])

        # Pass forward decoupling module
        ts_tokens, tab_tokens_unique, tab_tokens_shared = self.decoupling_module(ts_tokens, tab_tokens)

        if output_intermediate:
            return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)

        # x = torch.cat([ts_tokens, tab_tokens_unique, tab_tokens_shared, cls_tokens.unsqueeze(1)], dim=1)
        x = torch.cat([tab_tokens_unique, cls_tokens.unsqueeze(1)], dim=1)
        x_context = torch.cat([tab_tokens_shared, ts_tokens,], dim=1)

        for layer_idx, block in enumerate(self.blocks):
            if "cross_attention" not in block:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

                if self.unique_cross_block and layer_idx < self.n_self_blocks-1:
                    block = self.unique_cross_block

                    block = cast(nn.ModuleDict, block)

                    # Normalize the tokens from both modalities if prenormalization is enabled
                    x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                    x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                    # Forward pass through the cross-attention block
                    x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                    # Residual connections after the attention layer for both modalities
                    x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                    x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                    # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                    x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                    x_residual = block["cross_ffn"](x_residual)
                    x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                    x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                    x_context_residual = block["cross_ffn"](x_context_residual)
                    x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention")

            else:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                # Forward pass through the cross-attention block
                x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                x_residual = block["cross_ffn"](x_residual)
                x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                x_context_residual = block["cross_ffn"](x_context_residual)
                x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention") 

        # Concatenation is only for hook purposes
        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x
        # if output_intermediate:
        #     ts_tokens = output_tensor[:, self.n_tabular_attrs:self.n_tabular_attrs+self.n_time_series_attrs]
        #     tab_tokens_unique = output_tensor[:, self.n_tabular_attrs+self.n_time_series_attrs:-1]
        #     tab_tokens_shared = output_tensor[:, :self.n_tabular_attrs]
        #     return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)
        return output_tensor

class FT_Interleaved_2UniFTs_Inverted_TS(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        freeze_self_attention: bool = False,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
        tabular_unimodal_encoder: str,
        ts_unimodal_encoder: str,
        intermediate_mode: str = "average",
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks

        self.tabular_lin_proj = nn.Linear(d_token, 2*d_token)
        self.time_series_lin_proj = nn.Linear(d_token, d_token)

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        self.tabular_unimodal_encoder = get_nn_module(tabular_unimodal_encoder)
        self.ts_unimodal_encoder = get_nn_module(ts_unimodal_encoder)

        self.intermediate_mode = intermediate_mode
        
        layers = []
        total_blocks = max(self.n_self_blocks, self.n_cross_blocks)

        if self.n_cross_blocks > 1:
            for layer_idx in range(total_blocks):
                if layer_idx < self.n_self_blocks:
                    layers.append(self._init_attention_block(layer_idx))
                
                if layer_idx < self.n_cross_blocks:
                    layers.append(self._init_cross_attention_block(layer_idx))
        elif self.n_cross_blocks == 1:
            for layer_idx in range(self.n_self_blocks):
                layers.append(self._init_attention_block(layer_idx))
            self.unique_cross_block = self._init_cross_attention_block(0)

        self.blocks = nn.ModuleList(layers)

        if freeze_self_attention:
            self.freeze_self_attention()


    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                f"attention": MultiheadAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                f"ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
        layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                "cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                "cross_ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                "cross_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer["cross_attention_normalization"] = get_nn_module(self.attention_normalization)
        layer["cross_ffn_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def freeze_self_attention(self):
        for block in self.blocks:
            if "attention" in block:
                block["attention"].requires_grad_(False)
                if "attention_normalization" in block:
                    block["attention_normalization"].requires_grad_(False)

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "bidirectional_attention",
                    "bidirectional_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None, output_intermediate: bool = False) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        # Extract the class tokens
        cls_tokens = x[:, -1, :]

        # Unimodal encoding for both modalities
        ts_tokens = self.ts_unimodal_encoder(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_unimodal_encoder(x[:, self.n_time_series_attrs :-1])

        # Linear projections for alignment
        ts_tokens = self.time_series_lin_proj(ts_tokens)
        tab_tokens = self.tabular_lin_proj(tab_tokens)
        
        tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,::2,:]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,1::2,:]

        if output_intermediate:
            # return aggregate_tokens(ts_tokens, tab_tokens_unique, tab_tokens_shared, mode=self.intermediate_mode)
            return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)

        # x = torch.cat([ts_tokens, tab_tokens_unique, tab_tokens_shared, cls_tokens.unsqueeze(1)], dim=1)
        x = torch.cat([tab_tokens_unique, cls_tokens.unsqueeze(1)], dim=1)
        x_context = torch.cat([tab_tokens_shared, ts_tokens,], dim=1)

        for layer_idx, block in enumerate(self.blocks):
            if "cross_attention" not in block:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

                if self.unique_cross_block and layer_idx < self.n_self_blocks-1:
                    block = self.unique_cross_block

                    block = cast(nn.ModuleDict, block)

                    # Normalize the tokens from both modalities if prenormalization is enabled
                    x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                    x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                    # Forward pass through the cross-attention block
                    x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                    # Residual connections after the attention layer for both modalities
                    x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                    x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                    # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                    x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                    x_residual = block["cross_ffn"](x_residual)
                    x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                    x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                    x_context_residual = block["cross_ffn"](x_context_residual)
                    x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention")

            else:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                # Forward pass through the cross-attention block
                x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                x_residual = block["cross_ffn"](x_residual)
                x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                x_context_residual = block["cross_ffn"](x_context_residual)
                x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention") 

        # Concatenation is only for hook purposes
        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x
        # if output_intermediate:
        #     ts_tokens = output_tensor[:, self.n_tabular_attrs:self.n_tabular_attrs+self.n_time_series_attrs]
        #     tab_tokens_unique = output_tensor[:, self.n_tabular_attrs+self.n_time_series_attrs:-1]
        #     tab_tokens_shared = output_tensor[:, :self.n_tabular_attrs]
        #     return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)
        return output_tensor

class FT_Interleaved_2UniFTs_nodecoupling(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        freeze_self_attention: bool = False,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
        tabular_unimodal_encoder: str,
        ts_unimodal_encoder: str,
        intermediate_mode: str = "average",
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks

        self.time_series_lin_proj = nn.Linear(d_token, d_token)
        self.tabular_lin_proj = nn.Linear(d_token, 2*d_token)

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        self.tabular_unimodal_encoder = get_nn_module(tabular_unimodal_encoder)
        self.ts_unimodal_encoder = get_nn_module(ts_unimodal_encoder)

        self.intermediate_mode = intermediate_mode
        
        layers = []
        total_blocks = max(self.n_self_blocks, self.n_cross_blocks)

        if self.n_cross_blocks > 1:
            for layer_idx in range(total_blocks):
                if layer_idx < self.n_self_blocks:
                    layers.append(self._init_attention_block(layer_idx))
                
                if layer_idx < self.n_cross_blocks:
                    layers.append(self._init_cross_attention_block(layer_idx))
        elif self.n_cross_blocks == 1:
            for layer_idx in range(self.n_self_blocks):
                layers.append(self._init_attention_block(layer_idx))
            self.unique_cross_block = self._init_cross_attention_block(0)

        self.blocks = nn.ModuleList(layers)

        if freeze_self_attention:
            self.freeze_self_attention()


    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                f"attention": MultiheadAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                f"ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
        layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                "cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                "cross_ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                "cross_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer["cross_attention_normalization"] = get_nn_module(self.attention_normalization)
        layer["cross_ffn_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def freeze_self_attention(self):
        for block in self.blocks:
            if "attention" in block:
                block["attention"].requires_grad_(False)
                if "attention_normalization" in block:
                    block["attention_normalization"].requires_grad_(False)

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "bidirectional_attention",
                    "bidirectional_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None, output_intermediate: bool = False) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        # Extract the class tokens
        cls_tokens = x[:, -1, :]

        # Unimodal encoding for both modalities
        ts_tokens = self.ts_unimodal_encoder(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_unimodal_encoder(x[:, self.n_time_series_attrs :-1])

        if output_intermediate:
            # return aggregate_tokens(ts_tokens, tab_tokens_unique, tab_tokens_shared, mode=self.intermediate_mode)
            return  ts_tokens.mean(dim=1), tab_tokens.mean(dim=1), None

        # x = torch.cat([ts_tokens, tab_tokens_unique, tab_tokens_shared, cls_tokens.unsqueeze(1)], dim=1)
        x = torch.cat([tab_tokens, cls_tokens.unsqueeze(1)], dim=1)
        x_context = ts_tokens

        for layer_idx, block in enumerate(self.blocks):
            if "cross_attention" not in block:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

                if self.unique_cross_block and layer_idx < self.n_self_blocks-1:
                    block = self.unique_cross_block

                    block = cast(nn.ModuleDict, block)

                    # Normalize the tokens from both modalities if prenormalization is enabled
                    x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                    x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                    # Forward pass through the cross-attention block
                    x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                    # Residual connections after the attention layer for both modalities
                    x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                    x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                    # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                    x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                    x_residual = block["cross_ffn"](x_residual)
                    x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                    x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                    x_context_residual = block["cross_ffn"](x_context_residual)
                    x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention")

            else:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                # Forward pass through the cross-attention block
                x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                x_residual = block["cross_ffn"](x_residual)
                x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                x_context_residual = block["cross_ffn"](x_context_residual)
                x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention") 

        # Concatenation is only for hook purposes
        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x
        # if output_intermediate:
        #     ts_tokens = output_tensor[:, self.n_tabular_attrs:self.n_tabular_attrs+self.n_time_series_attrs]
        #     tab_tokens_unique = output_tensor[:, self.n_tabular_attrs+self.n_time_series_attrs:-1]
        #     tab_tokens_shared = output_tensor[:, :self.n_tabular_attrs]
        #     return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)
        return output_tensor

class FT_Interleaved_2UniFTs_DoubleTok(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        freeze_self_attention: bool = False,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
        tabular_unimodal_encoder: str,
        tabular_shared_encoder: str,
        ts_unimodal_encoder: str,
        intermediate_mode: str = "average",
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks

        self.linear_proj = nn.Linear(d_token, d_token)

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        self.tabular_unimodal_encoder = get_nn_module(tabular_unimodal_encoder)
        self.tabular_shared_encoder = get_nn_module(tabular_shared_encoder)
        self.ts_unimodal_encoder = get_nn_module(ts_unimodal_encoder)

        self.intermediate_mode = intermediate_mode
        
        layers = []
        total_blocks = max(self.n_self_blocks, self.n_cross_blocks)

        if self.n_cross_blocks > 1:
            for layer_idx in range(total_blocks):
                if layer_idx < self.n_self_blocks:
                    layers.append(self._init_attention_block(layer_idx))
                
                if layer_idx < self.n_cross_blocks:
                    layers.append(self._init_cross_attention_block(layer_idx))
        elif self.n_cross_blocks == 1:
            for layer_idx in range(self.n_self_blocks):
                layers.append(self._init_attention_block(layer_idx))
            self.unique_cross_block = self._init_cross_attention_block(0)

        self.blocks = nn.ModuleList(layers)

        if freeze_self_attention:
            self.freeze_self_attention()


    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                f"attention": MultiheadAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                f"ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
        layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                "cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                "cross_ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                "cross_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer["cross_attention_normalization"] = get_nn_module(self.attention_normalization)
        layer["cross_ffn_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def freeze_self_attention(self):
        for block in self.blocks:
            if "attention" in block:
                block["attention"].requires_grad_(False)
                if "attention_normalization" in block:
                    block["attention_normalization"].requires_grad_(False)

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "bidirectional_attention",
                    "bidirectional_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(
        self,
        ts_tokens: Tensor,
        tab_tokens_unique: Tensor,
        tab_tokens_shared: Tensor,
        output_intermediate: bool = False
    ) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """
        
        # if x.ndim != 3:
        #     raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        # if x_context is not None and x_context.ndim != 3:
        #     raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        # if x_context is not None and x.shape[-1] != x_context.shape[-1]:
        #     raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = ts_tokens.shape[0] # Save the batch size for later use in explainability

        # Extract the class tokens
        cls_tokens = tab_tokens_shared[:, -1, :]

        # Unimodal encoding for both modalities
        ts_tokens = self.ts_unimodal_encoder(ts_tokens)
        tab_tokens_unique = self.tabular_unimodal_encoder(tab_tokens_unique)
        tab_tokens_shared = self.tabular_shared_encoder(tab_tokens_shared[:, :-1])

        if output_intermediate:
            ts_tokens = self.linear_proj(ts_tokens)
            tab_tokens_unique = self.linear_proj(tab_tokens_unique)
            tab_tokens_shared = self.linear_proj(tab_tokens_shared)
            return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)

        # x = torch.cat([ts_tokens, tab_tokens_unique, tab_tokens_shared, cls_tokens.unsqueeze(1)], dim=1)
        x = torch.cat([tab_tokens_unique, cls_tokens.unsqueeze(1)], dim=1)
        x_context = torch.cat([tab_tokens_shared, ts_tokens,], dim=1)

        for layer_idx, block in enumerate(self.blocks):
            if "cross_attention" not in block:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

                if self.unique_cross_block and layer_idx < self.n_self_blocks-1:
                    block = self.unique_cross_block

                    block = cast(nn.ModuleDict, block)

                    # Normalize the tokens from both modalities if prenormalization is enabled
                    x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                    x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                    # Forward pass through the cross-attention block
                    x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                    # Residual connections after the attention layer for both modalities
                    x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                    x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                    # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                    x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                    x_residual = block["cross_ffn"](x_residual)
                    x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                    x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                    x_context_residual = block["cross_ffn"](x_context_residual)
                    x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention")

            else:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                # Forward pass through the cross-attention block
                x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                x_residual = block["cross_ffn"](x_residual)
                x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                x_context_residual = block["cross_ffn"](x_context_residual)
                x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention") 

        # Concatenation is only for hook purposes
        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x

        return output_tensor

class FT_Interleaved_Inverted(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        freeze_self_attention: bool = False,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks

        self.tabular_lin_proj = nn.Linear(d_token, 2*d_token)
        self.time_series_lin_proj = nn.Linear(d_token, d_token)

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        
        layers = []
        total_blocks = max(self.n_self_blocks, self.n_cross_blocks)

        if self.n_cross_blocks > 1:
            for layer_idx in range(total_blocks):
                if layer_idx < self.n_self_blocks:
                    layers.append(self._init_attention_block(layer_idx))
                
                if layer_idx < self.n_cross_blocks:
                    layers.append(self._init_cross_attention_block(layer_idx))
        elif self.n_cross_blocks == 1:
            for layer_idx in range(self.n_self_blocks):
                layers.append(self._init_attention_block(layer_idx))
            self.unique_cross_block = self._init_cross_attention_block(0)

        self.blocks = nn.ModuleList(layers)

        if freeze_self_attention:
            self.freeze_self_attention()


    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                f"attention": MultiheadAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                f"ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
        layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                "cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                "cross_ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                "cross_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer["cross_attention_normalization"] = get_nn_module(self.attention_normalization)
        layer["cross_ffn_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def freeze_self_attention(self):
        for block in self.blocks:
            if "attention" in block:
                block["attention"].requires_grad_(False)
                if "attention_normalization" in block:
                    block["attention_normalization"].requires_grad_(False)

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "bidirectional_attention",
                    "bidirectional_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None, output_intermediate: bool = False) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        # Extract the class tokens
        cls_tokens = x[:, -1, :]

        # Linear projections for alignment
        ts_tokens = self.time_series_lin_proj(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_lin_proj(x[:, self.n_time_series_attrs :-1])
        
        tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,::2,:]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,1::2,:]

        if output_intermediate:
            return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)

        # x = torch.cat([ts_tokens, tab_tokens_unique, tab_tokens_shared, cls_tokens.unsqueeze(1)], dim=1)
        x = torch.cat([tab_tokens_unique, tab_tokens_shared, cls_tokens.unsqueeze(1)], dim=1)
        # x_context = torch.cat([tab_tokens_shared, ts_tokens,], dim=1)
        x_context = ts_tokens

        for layer_idx, block in enumerate(self.blocks):
            if "cross_attention" not in block:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

                if self.unique_cross_block and layer_idx < self.n_self_blocks-1:
                    block = self.unique_cross_block

                    block = cast(nn.ModuleDict, block)

                    # Normalize the tokens from both modalities if prenormalization is enabled
                    x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                    x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                    # Forward pass through the cross-attention block
                    x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                    # Residual connections after the attention layer for both modalities
                    x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                    x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                    # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                    x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                    x_residual = block["cross_ffn"](x_residual)
                    x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                    x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                    x_context_residual = block["cross_ffn"](x_context_residual)
                    x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention")

            else:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                # Forward pass through the cross-attention block
                x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                x_residual = block["cross_ffn"](x_residual)
                x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                x_context_residual = block["cross_ffn"](x_context_residual)
                x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention") 

        # Concatenation is only for hook purposes
        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x

        return output_tensor

class FT_Interleaved_2UniFTs_Dummy(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        freeze_self_attention: bool = False,
        n_tabular_attrs: int,
        n_time_series_attrs: int,
        tabular_unimodal_encoder: str,
        ts_unimodal_encoder: str,
        intermediate_mode: str = "average",
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks

        self.tabular_lin_proj = nn.Linear(d_token, 2*d_token)
        self.time_series_lin_proj = nn.Linear(d_token, d_token)

        self.n_tabular_attrs = n_tabular_attrs
        self.n_time_series_attrs = n_time_series_attrs

        self.tabular_unimodal_encoder = get_nn_module(tabular_unimodal_encoder)
        self.ts_unimodal_encoder = get_nn_module(ts_unimodal_encoder)

        self.intermediate_mode = intermediate_mode
        
        layers = []
        total_blocks = max(self.n_self_blocks, self.n_cross_blocks)

        if self.n_cross_blocks > 1:
            for layer_idx in range(total_blocks):
                if layer_idx < self.n_self_blocks:
                    layers.append(self._init_attention_block(layer_idx))
                
                if layer_idx < self.n_cross_blocks:
                    layers.append(self._init_cross_attention_block(layer_idx))
            self.unique_cross_block = None
        elif self.n_cross_blocks == 1:
            for layer_idx in range(self.n_self_blocks):
                layers.append(self._init_attention_block(layer_idx))
            self.unique_cross_block = self._init_cross_attention_block(0)

        self.blocks = nn.ModuleList(layers)

        if freeze_self_attention:
            self.freeze_self_attention()


    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                f"attention": MultiheadAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                f"ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
        layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
                "cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                "cross_ffn": self.FFN(
                    d_token=self.d_token,
                    d_hidden=self.ffn_d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation,
                ),
                "cross_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
            }
        )
        if layer_idx or not self.prenormalization or self.first_prenormalization:
            layer["cross_attention_normalization"] = get_nn_module(self.attention_normalization)
        layer["cross_ffn_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def freeze_self_attention(self):
        for block in self.blocks:
            if "attention" in block:
                block["attention"].requires_grad_(False)
                if "attention_normalization" in block:
                    block["attention_normalization"].requires_grad_(False)

    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "cross_attention",
                    "cross_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "bidirectional_attention",
                    "bidirectional_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None, output_intermediate: bool = False) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        # Extract the class tokens
        cls_tokens = x[:, -1, :]

        # Unimodal encoding for both modalities
        ts_tokens = self.ts_unimodal_encoder(x[:, : self.n_time_series_attrs])
        tab_tokens = self.tabular_unimodal_encoder(x[:, self.n_time_series_attrs :-1])

        # Linear projections for alignment
        ts_tokens = self.time_series_lin_proj(ts_tokens)
        tab_tokens = self.tabular_lin_proj(tab_tokens)
        
        tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,::2,:]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:,1::2,:]

        if output_intermediate:
            # return aggregate_tokens(ts_tokens, tab_tokens_unique, tab_tokens_shared, mode=self.intermediate_mode)
            return ts_tokens.mean(dim=1), tab_tokens_unique.mean(dim=1), tab_tokens_shared.mean(dim=1)

        # x = torch.cat([ts_tokens, tab_tokens_unique, tab_tokens_shared, cls_tokens.unsqueeze(1)], dim=1)
        x = torch.cat([tab_tokens_unique, cls_tokens.unsqueeze(1)], dim=1)
        x_context = torch.cat([tab_tokens_shared, ts_tokens,], dim=1)

        for layer_idx, block in enumerate(self.blocks):
            if "cross_attention" not in block:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

                if self.unique_cross_block and layer_idx < self.n_self_blocks-1:
                    block = self.unique_cross_block

                    block = cast(nn.ModuleDict, block)

                    # Normalize the tokens from both modalities if prenormalization is enabled
                    x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                    x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                    # Forward pass through the cross-attention block
                    x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                    # Residual connections after the attention layer for both modalities
                    x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                    x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                    # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                    x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                    x_residual = block["cross_ffn"](x_residual)
                    x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                    x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                    x_context_residual = block["cross_ffn"](x_context_residual)
                    x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention")

            else:
                block = cast(nn.ModuleDict, block)

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "cross_attention", x, stage="cross_attention")
                x_context_residual = self._start_residual(block, "cross_attention", x_context, stage="cross_attention")

                # Forward pass through the cross-attention block
                x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "cross_attention", x, x_residual, stage="cross_attention")
                x_context = self._end_residual(block, "cross_attention", x_context, x_context_residual, stage="cross_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "cross_ffn", x, stage="cross_attention")
                x_residual = block["cross_ffn"](x_residual)
                x = self._end_residual(block, "cross_ffn", x, x_residual, stage="cross_attention")

                x_context_residual = self._start_residual(block, "cross_ffn", x_context, stage="cross_attention")
                x_context_residual = block["cross_ffn"](x_context_residual)
                x_context = self._end_residual(block, "cross_ffn", x_context, x_context_residual, stage="cross_attention") 

        # Concatenation is only for hook purposes
        output_tensor = torch.cat([x_context, x], dim=1) if x_context is not None else x
        return output_tensor