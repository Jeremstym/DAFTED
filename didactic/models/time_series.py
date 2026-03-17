from typing import Any, Dict, List, Optional, Sequence

import torch
from dataprocessing.data.orchid.datapipes import filter_time_series_attributes
from torch import Tensor, nn
from torch.nn import functional as F


def differentiate_ts(x: Tensor, order: int = 1) -> Tensor:
    """Differentiates the input time series tensor.

    Args:
        x: (N, resample_dim), Input time series tensor.
        order: Order of the differentiation.

    Returns:
        (N, resample_dim - order), Differentiated time series tensor.
    """
    tensor = x.diff(dim=-1, n=order)
    return tensor


def multi_differentiate_ts(x: Tensor, orders: Sequence[int]) -> List[Tensor]:
    """Differentiates the input time series tensor multiple times.

    Args:
        x: (N, resample_dim), Input time series tensor.
        orders: Orders of the differentiations.

    Returns:
        (N, resample_dim - len(orders)), Differentiated time series tensor.
    """
    tensor = x.unsqueeze(1).repeat(1, len(orders), 1)
    tensor_list = []
    for i, order in enumerate(orders):
        sub_tensor = differentiate_ts(tensor[:, i], order)
        tensor_list.append(sub_tensor)
    return tensor_list


def ts_collapse_on_features(ts: Tensor, n_time_series_attrs: int) -> Tensor:
    """Collapses the time series tensor on the feature dimension, each line
    split into rows (Num items * num features, time steps).
    Args:
        ts: (N, 1, S_ts * num_time_units), Input time series tensor.
        n_time_series_attrs: Number of features in the time series.
    Returns:
        (N * num_features, 1, S_ts), Collapsed time series tensor.
    """
    ts_out = ts.reshape(ts.shape[0] * n_time_series_attrs, 1, -1)  # (N * num_features, 1, S_ts)
    return ts_out

class TimeSeriesFeatureFusion(nn.Module):
    """Feature fusion for time series."""

    def __init__(self, n_time_series_attrs: int, fusion_method: str = "linear") -> None:
        """Initializes class instance.
        Args:
            n_time_series_attrs: Number of time series attributes := F.
            fusion_method: Fusion method to use. Can be one of ["linear"].
        """        
        super().__init__()
        self.n_time_series_attrs = n_time_series_attrs
        self.fusion_method = fusion_method

        self.fc = nn.Linear(n_time_series_attrs, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Fuses the input time series tensor.

        Args:
            x: (S, B, F*T), Input time series tensor.

        Returns:
            (N, S, E), Fused time series tensor.
        """
        x = x.to(self.fc.weight.device)
        if self.fusion_method == "linear":
            x = x.reshape(-1, x.shape[1], self.n_time_series_attrs, x.shape[-1] // self.n_time_series_attrs) # (S, B, F, T)
            x = x.transpose(-1, -2) # (S, B, T, F)
            x = self.fc(x) # (S, B, T, 1)
            x = x.squeeze(-1) # (S, B, T)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return x


class MultiLinearEmbedding(nn.Module):
    """Multi-linear embedding for time series."""

    def __init__(self, in_features: int, n_sub_ts: int, d_model: int) -> None:
        """Initializes class instance.

        Args:
            n_sub_ts: Number of sub-time series.
            d_model: Token dimensionality.
        """
        super().__init__()
        self.in_features = in_features
        self.n_sub_ts = n_sub_ts
        self.d_model = d_model

        for i in range(n_sub_ts):
            setattr(self, f"linear_{i}", nn.Linear(self.in_features - i, d_model))

    def forward(self, x_list: List[Tensor]) -> Tensor:
        """Embeds the input time series tensor.

        Args:
            x: (N, S, E), Input time series tensor.

        Returns:
            (N, S, E), Embedded time series tensor.
        """
        # x = x.permute(1, 0, 2)
        # x = torch.stack([getattr(self, f"linear_{i}")(x[i]) for i in range(self.n_sub_ts)], dim=0)
        # x = x.permute(1, 0, 2)
        # return x.mean(dim=1)

        for i in range(self.n_sub_ts):
            x_list[i] = getattr(self, f"linear_{i}")(x_list[i])

        x_stacked = torch.stack(x_list, dim=1)

        return torch.mean(x_stacked, dim=1)


class TimeSeriesPositionalEncoding(nn.Module):
    def __init__(self, n_positions: int, d_model: int) -> None:
        """Initializes class instance.

        Args:
            d_model: Token dimensionality.
        """
        super().__init__()
        self.n_positions = n_positions
        self.d_model = d_model

    # Define specific positional encoding functions for time series
    # From HuggingFace's implementation of the Transformer model

    @staticmethod
    def angle_defn(pos, i, d_model):
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        return pos * angle_rates

    def positional_encoding(self, position, d_model, dtype=torch.float32):
        # create the sinusoidal pattern for the positional encoding
        angle_rads = self.angle_defn(
            torch.arange(position, dtype=torch.int64).to(dtype).unsqueeze(1),
            torch.arange(d_model, dtype=torch.int64).to(dtype).unsqueeze(0),
            d_model,
        )

        sines = torch.sin(angle_rads[:, 0::2])
        cosines = torch.cos(angle_rads[:, 1::2])

        pos_encoding = torch.cat([sines, cosines], dim=-1)
        return pos_encoding

    def forward(self, x: Tensor) -> Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x: (N, S, E), Input tensor to add positional encoding to.

        Returns:
            (N, S, E), Input tensor with positional encoding added.
        """
        # Create the positional encoding
        pos_enc = self.positional_encoding(self.n_positions, self.d_model, dtype=x.dtype)
        pos_enc = pos_enc.unsqueeze(0)  # (1, S, E)
        pos_enc = pos_enc.to(x.device)

        # Add the positional encoding to the input tensor
        x = x + pos_enc

        return x


class TimeSeriesEmbedding(nn.Module):
    """Embedding for time series which resamples the time dim and/or passes through an arbitrary learnable model."""

    def __init__(
        self,
        resample_dim: int,
        model: Optional[nn.Module] = None,
        corruption_rate: float = 0.0,
        pooling: bool = False,
        channel_first: bool = False,
    ) -> None:
        """Initializes class instance.

        Args:
            resample_dim: Target size for an interpolation resampling of the time series.
            model: Model that learns to embed the time series. If not provided, no projection is learned and the
                embedding is simply the resampled time series. The model should take as input a tensor of shape
                (N, `resample_dim`) and output a tensor of shape (N, E), where E is the embedding size.
        """
        super().__init__()
        self.model = model
        self.resample_dim = resample_dim
        self.corruption_rate = corruption_rate
        self.pooling = pooling
        self.channel_first = channel_first

    def corrupt(self, x: Tensor, corruption_rate: float = 0.1) -> Tensor:
        """Corrupts the input tensor by setting a fraction of its values to zero.

        Args:
            x: (N, S, E), Input tensor to corrupt.
            corruption_rate: Fraction of values to set to zero.

        Returns:
            (N, S, E), Corrupted input tensor.
        """
        mask = torch.rand_like(x) < corruption_rate
        return x * mask

    def sequence_pooling_ts(self, x: Tensor, pooling: str = "mean") -> Tensor:
        """Applies a pooling operation along the time dimension of the input tensor.

        Args:
            x: (N, S, E), Input tensor to pool.
            pooling: Pooling operation to apply. Can be one of ["mean", "max"].

        Returns:
            (N, E), Pooled input tensor.
        """
        if pooling == "mean":
            return x.mean(dim=1)
        elif pooling == "max":
            return x.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling operation")

    def forward(self, time_series: Dict[Any, Tensor] | Sequence[Tensor]) -> Tensor:
        """Stacks the time series, optionally 1) resampling them and/or 2) projecting them to a target embedding.

        Args:
            time_series: (K: S, V: (N, ?)) or S * (N, ?): Time series batches to embed, where the dimensionality of each
                time series can vary.

        Returns:
            (N, S, E), Embedding of the time series.
        """
        if not isinstance(time_series, dict):
            time_series = {idx: t for idx, t in enumerate(time_series)}

        # Resample time series to make sure all of them are of `resample_dim`
        for t_id, t in time_series.items():
            if t.shape[-1] != self.resample_dim:
                # Temporarily reshape time series batch tensor to be 3D to be able to use torch's interpolation
                # (N, ?) -> (N, `resample_dim`)
                time_series[t_id] = F.interpolate(t.unsqueeze(1), size=self.resample_dim, mode="linear").squeeze(dim=1)

        # Extract the time series from the dictionary and stack them along the batch dimension
        ts_values = list(time_series.values())  # (S, N, `resample_dim`)
        # Check indices of nan values
        # print(f"nan indices: {torch.isnan(torch.stack(x, dim=1)).all(dim=2).nonzero()}")
        ts_values = [torch.nan_to_num(attr, nan=0.0) for attr in ts_values]  # Replace NaN values with 0.0
        if self.channel_first:
            x = torch.stack(ts_values, dim=2)  # (S, N, `resample_dim`) -> (N, `resample_dim`, S)
            if self.model:
                x = self.model(x)
        elif self.model:
            # If provided with a learnable model, use it to predict the embedding of each time series separately
            model_outputs = [self.model(attr) for attr in ts_values]  # (S, N, `resample_dim`) -> (S, N, E)
            # Stack the embeddings of all the time series (along the batch dimension) to make only one tensor
            x = torch.stack(model_outputs, dim=1)  # (S, N, E) -> (N, S, E)

        if self.corruption_rate:
            x = self.corrupt(x, self.corruption_rate)

        if self.pooling:
            x = self.sequence_pooling_ts(x)  # (N, S, E) -> (N, E)

        if x.ndim == 4:
            n, s, s_ts, e = x.shape
            x = x.view(n, s * s_ts, e)

        return x
