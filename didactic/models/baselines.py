import math
from typing import Callable, Dict, Literal, Tuple, Union

import hydra
import torch
from IRENE.models.configs import get_IRENE_config
from IRENE.models.encoder import Encoder as IRENEncoder
from omegaconf import DictConfig
from scipy.special import binom, factorial
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init

# import didactic.models.transformer
from didactic.models.layers import _QKVLinearProjection, _QKVMatrixMultiplication

from didactic.models.layers import get_nn_module

ModuleType = Union[str, Callable[..., nn.Module]]


class BidirectionalMultimodalAttention(nn.Module):
    """Bidirectional Multimodal Attention."""

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool = True,
        initialization: Literal["kaiming", "xavier"] = "kaiming",
    ):
        """Initializes class instance.

        Args:
            d_token: Token size. Must be a multiple of `n_heads`.
            n_heads: Number of attention heads. If greater than 1, then the module will have an additional output layer
                (so called "mixing" layer).
            dropout: Dropout rate for the attention map. The dropout is applied to *probabilities* and does not affect
                logits.
            bias: If `True`, then input (and output, if presented) layers also have bias.
            initialization: Initialization for input projection layers. Must be one of ['kaiming', 'xavier'].
        """
        super().__init__()
        self.mod0_linear_proj = _QKVLinearProjection(d_token, n_heads, bias=bias, initialization=initialization)
        self.mod1_linear_proj = _QKVLinearProjection(d_token, n_heads, bias=bias, initialization=initialization)
        self.mod0_self_mat_mul = _QKVMatrixMultiplication(d_token, n_heads, dropout, bias=bias)
        self.mod1_self_mat_mul = _QKVMatrixMultiplication(d_token, n_heads, dropout, bias=bias)
        self.mod0_cross_mat_mul = _QKVMatrixMultiplication(d_token, n_heads, dropout, bias=bias)
        self.mod1_cross_mat_mul = _QKVMatrixMultiplication(d_token, n_heads, dropout, bias=bias)

    def forward(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass through the attention operations.

        Args:
            x0: (N, S_0, E), Tokens from 1st modality.
            x1: (N, S_1, E), Tokens from 2nd modality.

        Returns:
            (N, S_0, E) + (N, S_1, E), attention output tokens for both modalities.
        """
        q0, k0, v0 = self.mod0_linear_proj(x0, x0)
        q1, k1, v1 = self.mod1_linear_proj(x1, x1)

        self_0, _ = self.mod0_self_mat_mul(q0, k0, v0)
        cross_0, _ = self.mod0_cross_mat_mul(q0, k1, v1)
        self_1, _ = self.mod1_self_mat_mul(q1, k1, v1)
        cross_1, _ = self.mod1_cross_mat_mul(q1, k0, v0)

        x0 = self_0 + cross_0
        x1 = self_1 + cross_1

        return x0, x1


class MLP(nn.Module):
    """A simple MLP for tabular data."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_layers: int = 2,
        d_token: int = 192,
        dropout: float = 0.1,
    ) -> None:
        """Initializes class instance.

        Args:
            in_features: the number of input features.
            n_layers: the number of hidden layers.
            d_token: the number of hidden units in each layer.
            dropout: the dropout rate.
        """
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features if i == 0 else d_token, d_token) for i in range(n_layers)])
        self.head = nn.Linear(d_token, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass.

        Args:
            x: the input tensor.

        Returns:
            the output tensor.
        """
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.head(x)


class FlatConcatMLP(nn.Module):
    """A MPL to encode the concatenated input after the tabular Transformer."""

    def __init__(self, sequence_length: int, n_mlp_layers: int = 2, d_token=192, dropout=0.1) -> None:
        """Initializes class instance.

        Args:
            tab_tokens: the number of tokens from the tabular Transformer.
            ts_feature: the number of features from the time series Transformer.
        """
        super().__init__()

        self.mlp = MLP(
            sequence_length * d_token, out_features=d_token, n_layers=n_mlp_layers, d_token=d_token, dropout=dropout
        )

    def forward(self, tab_tokens: Tensor, ts_tokens: Tensor) -> Tensor:
        """Performs the forward pass.

        Args:
            tab_tokens: the tabular tokens.
            ts_feature: the time series feature.

        Returns:
            the output tensor.
        """
        tab_tokens = tab_tokens.flatten(start_dim=1)
        ts_tokens = ts_tokens.flatten(start_dim=1)
        x = torch.cat((tab_tokens, ts_tokens), dim=1)  # (N, S*E)
        output = self.mlp(x)  # (N, E)
        return output.unsqueeze(1)  # (N, 1, E)


class ConcatMLP(nn.Module):
    """A MPL to encode the concatenated input after the tabular Transformer."""

    def __init__(self, tabular_encoder: str, n_mlp_layers: int = 2, d_token=192, dropout=0.1) -> None:
        """Initializes class instance.

        Args:
            tab_tokens: the number of tokens from the tabular Transformer.
            ts_feature: the number of features from the time series Transformer.
        """
        super().__init__()

        self.tabular_encoder = get_nn_module(tabular_encoder)
        self.mlp = MLP(2 * d_token, out_features=d_token, n_layers=n_mlp_layers, d_token=d_token, dropout=dropout)

    def forward(self, tab_tokens: Tensor, ts_tokens: Tensor) -> Tensor:
        """Performs the forward pass.

        Args:
            tab_tokens: the tabular tokens.
            ts_feature: the time series feature.

        Returns:
            the output tensor.
        """
        tabular_output = self.tabular_encoder(tab_tokens)
        # if isinstance(self.tabular_encoder, didactic.models.transformer.FT_Transformer):
        #     tabular_output = tabular_output[:, -1, :]  # Get the CLS token
        x = torch.cat((tabular_output, ts_tokens.mean(dim=1)), dim=1)  # (N, 2*E)
        output = self.mlp(x)  # (N, E)
        return output.unsqueeze(1)  # (N, 1, E)


class MMCLEncoder(nn.Module):
    """A MPL to encode the concatenated input after the tabular Transformer."""

    def __init__(
        self,
        n_tabular_attrs: int,
        ts_encoder: str,
        n_tab_mlp_layers: int = 2,
        tab_mlp_hidden: int = 192,
        n_mlp_layers: int = 2,
        d_token=192,
        dropout=0.1,
    ) -> None:
        """Initializes class instance.

        Args:
            tab_tokens: the number of tokens from the tabular Transformer.
            ts_feature: the number of features from the time series Transformer.
        """
        super().__init__()

        self.tabular_encoder = MLP(
            in_features=n_tabular_attrs,
            out_features=d_token,
            n_layers=n_tab_mlp_layers,
            d_token=tab_mlp_hidden,
            dropout=dropout,
        )
        self.ts_encoder = get_nn_module(ts_encoder)
        self.fusion_mlp = MLP(
            2 * d_token, out_features=d_token, n_layers=n_mlp_layers, d_token=d_token, dropout=dropout
        )

    def forward(
        self, tab_tokens: Tensor, ts_tokens: Tensor, output_intermediate: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor | None]:
        """Performs the forward pass.

        Args:
            tab_tokens: the tabular tokens.
            ts_feature: the time series feature.

        Returns:
            the output tensor.
        """
        tab_tokens = tab_tokens[:, :, 0]  # Select the first dimension as they are all the same
        tabular_output = self.tabular_encoder(tab_tokens)
        ts_output = self.ts_encoder(ts_tokens)
        if output_intermediate:
            return ts_output.mean(dim=1), tabular_output, None
        x = torch.cat((tabular_output, ts_tokens.mean(dim=1)), dim=1)  # (N, 2*E)
        output = self.fusion_mlp(x)  # (N, E)
        return output.unsqueeze(1)  # (N, 1, E)


class AvgConcatMLP(nn.Module):
    """A MPL to encode the concatenated input after the tabular Transformer."""

    def __init__(self, n_tabular_attrs: int, n_mlp_layers: int = 2, d_token=192, dropout=0.1) -> None:
        """Initializes class instance.

        Args:
            tab_tokens: the number of tokens from the tabular Transformer.
            ts_feature: the number of features from the time series Transformer.
        """
        super().__init__()

        self.mlp = MLP(
            n_tabular_attrs + d_token, out_features=d_token, n_layers=n_mlp_layers, d_token=d_token, dropout=dropout
        )

    def forward(self, tab_tokens: Tensor, ts_tokens: Tensor) -> Tensor:
        """Performs the forward pass.

        Args:
            tab_tokens: the tabular tokens.
            ts_feature: the time series feature.

        Returns:
            the output tensor.
        """
        tab_tokens = tab_tokens[:, :, 0]  # Select the first dimension as they are all the same
        x = torch.cat((tab_tokens, ts_tokens.mean(dim=1)), dim=1)  # (N, 2*E)
        output = self.mlp(x)  # (N, E)
        return output.unsqueeze(1)  # (N, 1, E)


class ConcatMLPDecoupling(nn.Module):
    """A MPL to encode the concatenated input after the tabular Transformer."""

    def __init__(self, n_tabular_attrs: int, n_mlp_layers: int = 2, d_token=192, dropout=0.1) -> None:
        """Initializes class instance.

        Args:
            tab_tokens: the number of tokens from the tabular Transformer.
            ts_feature: the number of features from the time series Transformer.
        """
        super().__init__()

        self.d_token = d_token
        self.n_tabular_attrs = n_tabular_attrs
        self.mlp = MLP(3 * d_token, out_features=d_token, n_layers=n_mlp_layers, d_token=d_token, dropout=dropout)
        self.tabular_lin_proj = nn.Linear(d_token, 2 * d_token)
        self.time_series_lin_proj = nn.Linear(d_token, d_token)
        # self.tabular_lin_proj = MLP(d_token, 2*d_token, n_layers=3, d_token=2*d_token, dropout=dropout)
        # self.time_series_lin_proj = MLP(d_token, d_token, n_layers=3, d_token=d_token, dropout=dropout)

    def forward(
        self, tab_tokens: Tensor, ts_tokens: Tensor, output_intermediate: bool = False
    ) -> tuple[Tensor, Tensor, Tensor | None] | Tensor:
        """Performs the forward pass.

        Args:
            tab_tokens: the tabular tokens.
            ts_feature: the time series feature.

        Returns:
            the output tensor.
        """

        ts_tokens = self.time_series_lin_proj(ts_tokens)
        tab_tokens = self.tabular_lin_proj(tab_tokens)
        tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:, ::2, :]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:, 1::2, :]

        # Average both modalities
        tab_tokens_unique = tab_tokens_unique.mean(dim=1)
        tab_tokens_shared = tab_tokens_shared.mean(dim=1)
        ts_tokens = ts_tokens.mean(dim=1)

        if output_intermediate:
            return ts_tokens, tab_tokens_unique, tab_tokens_shared

        # Set the unique tokens at the beginning of the sequence for extraction purposes
        x = torch.cat((tab_tokens_unique, tab_tokens_shared, ts_tokens), dim=1)
        output = self.mlp(x)
        return output.unsqueeze(1)  # (N, 1, E)


class ConcatMLPDecoupling2FTs(nn.Module):
    """A MPL to encode the concatenated input after the tabular Transformer."""

    def __init__(
        self,
        tabular_unimodal_encoder: str,
        ts_unimodal_encoder: str,
        n_tabular_attrs: int,
        n_mlp_layers: int = 2,
        d_token=192,
        dropout=0.1,
    ) -> None:
        """Initializes class instance.

        Args:
            tab_tokens: the number of tokens from the tabular Transformer.
            ts_feature: the number of features from the time series Transformer.
        """
        super().__init__()

        self.d_token = d_token
        self.n_tabular_attrs = n_tabular_attrs
        self.mlp = MLP(3 * d_token, out_features=d_token, n_layers=n_mlp_layers, d_token=d_token, dropout=dropout)
        self.tabular_unimodal_encoder = get_nn_module(tabular_unimodal_encoder)
        self.ts_unimodal_encoder = get_nn_module(ts_unimodal_encoder)
        self.tabular_lin_proj = nn.Linear(d_token, 2 * d_token)
        self.time_series_lin_proj = nn.Linear(d_token, d_token)

    def forward(
        self, tab_tokens: Tensor, ts_tokens: Tensor, output_intermediate: bool = False
    ) -> tuple[Tensor, Tensor, Tensor | None] | Tensor:
        """Performs the forward pass.

        Args:
            tab_tokens: the tabular tokens.
            ts_feature: the time series feature.

        Returns:
            the output tensor.
        """
        # Remove CLS token
        tab_tokens = tab_tokens[:, :-1, :]

        # Encode tabular and time series data
        tabular_output = self.tabular_unimodal_encoder(tab_tokens)
        ts_output = self.ts_unimodal_encoder(ts_tokens)

        # Linear projection
        ts_tokens = self.time_series_lin_proj(ts_output)
        tab_tokens = self.tabular_lin_proj(tabular_output)
        tab_tokens_unique = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:, ::2, :]
        tab_tokens_shared = tab_tokens.reshape(tab_tokens.shape[0], -1, self.d_token)[:, 1::2, :]

        # Average both modalities
        tabular_output_unique = tab_tokens_unique.mean(dim=1)
        tabular_output_shared = tab_tokens_shared.mean(dim=1)
        ts_output = ts_output.mean(dim=1)

        if output_intermediate:
            return ts_output, tabular_output_unique, tabular_output_shared

        # Set the unique outputs at the beginning of the sequence for extraction purposes
        x = torch.cat((tabular_output_unique, tabular_output_shared, ts_output), dim=1)
        output = self.mlp(x)
        return output.unsqueeze(1)  # (N, 1, E)


class TabularMLP(nn.Module):
    """A simple MLP for tabular data.

    The "MLP" module from "Revisiting Deep Learning Models for Tabular Data" by Gorishniy et al. (2021).
    The module is a simple MLP that can be used for tabular data.

    Notes:
        - This is a port of the `MLP` class from v0.0.13 of the `rtdl` package using the updated underlying `MLP`
          from v0.0.2 of the `rtdl_revisiting_models` package.

    References:
        - Original implementation is here:
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_layers: int = 2,
        d_token: int = 192,
        dropout: float = 0.1,
    ) -> None:
        """Initializes class instance.

        Args:
            in_features: the number of input features.
            n_layers: the number of hidden layers.
            d_token: the number of hidden units in each layer.
            dropout: the dropout rate.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features if i == 0 else d_token, d_token),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for i in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_token, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x: the tabular input tensor.

        Returns:
            the embedded output tensor.
        """
        x = x.mean(dim=2)  # (N, S_tab), the tensors are the same on the last dimension
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class IRENEModel(nn.Module):
    """A simple MLP for tabular data.

    The "MLP" module from "Revisiting Deep Learning Models for Tabular Data" by Gorishniy et al. (2021).
    The module is a simple MLP that can be used for tabular data.

    Notes:
        - This is a port of the `MLP` class from v0.0.13 of the `rtdl` package using the updated underlying `MLP`
          from v0.0.2 of the `rtdl_revisiting_models` package.

    References:
        - Original implementation is here:
    """

    def __init__(
        self,
    ) -> None:
        """Initializes class instance.

        Args:
            in_features: the number of input features.
            n_layers: the number of hidden layers.
            d_token: the number of hidden units in each layer.
            dropout: the dropout rate.
        """
        super().__init__()
        self.encoder = IRENEncoder(get_IRENE_config(), vis=False)

    def forward(self, tab_tokens: Tensor, ts_tokens: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x: the tabular input tensor.

        Returns:
            the embedded output tensor.
        """
        output_weights, _ = self.encoder(tab_tokens, ts_tokens)
        return output_weights.mean(dim=1).unsqueeze(1)


class IdentityEncoder(nn.Module):
    """An identity encoder that returns the input as output."""

    def __init__(self) -> None:
        """Initializes class instance."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass.

        Args:
            x: the input tensor.

        Returns:
            the output tensor, which is the same as the input.
        """
        return x
