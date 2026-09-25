import math
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from scipy.special import binom, factorial
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init

from tabpfn.model_loading import load_model_criterion_config

ModuleType = Union[str, Callable[..., nn.Module]]


def get_nn_module(module: ModuleType, *module_args, **module_kwargs) -> nn.Module:
    """Instantiates an ``nn.Module`` with the requested parameters.

    Args:
        module: Name of the ``nn.Module`` to instantiate, or function that initializes the module.
        *module_args: Positional arguments to pass to the ``nn.Module``'s constructor or generator function.
        **module_kwargs: Keyword arguments to pass to the ``nn.Module``'s constructor or generator function.

    Returns:
        Instance of the ``nn.Module``.
    """
    if callable(module):
        return module(*module_args, **module_kwargs)
    else:
        return getattr(nn, module)(*module_args, **module_kwargs)


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


def swiglu(x_context: Tensor) -> Tensor:
    """The SwiGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x_context.shape[-1] % 2 == 0
    a, b = x_context.chunk(2, dim=-1)
    return a * F.silu(b)


class ReGLU(nn.Module):
    """
    The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """
    The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class SwiGLU(nn.Module):
    """
    The SwiGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return swiglu(x)


class RMSNorm(nn.Module):
    """From Root Mean Square Layer Normalization https://arxiv.org/abs/1910.07467"""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class _QKVLinearProjection(nn.Module):
    def __init__(
        self, d_token: int, n_heads: int, bias: bool = True, initialization: Literal["kaiming", "xavier"] = "kaiming"
    ):
        """Initializes class instance.

        Args:
            d_token: Token size.
            n_heads: Number of attention heads. If equal to 1, then value projection matrix will always be initialized
                with Kaiming (regardless of `initialization` parameter), to follow torch.nn.MultiheadAttention.
            bias: If `True`, then input (and output, if presented) layers also have bias.
            initialization: Initialization for input projection layers. Must be one of ['kaiming', 'xavier'].
        """
        super().__init__()

        if initialization not in ["kaiming", "xavier"]:
            raise ValueError("`initialization` must be one of ['kaiming', 'xavier']")
        self.d_token = d_token
        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if V is directly used to compute output (i.e. not multi-head);
            # the latter one is initialized with Kaiming in torch
            if initialization == "xavier" and (m is not self.W_v or n_heads > 1):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes the query/key/value linear projections of tokens.

        Args:
            x_q: (N, S_q, E), Tokens from which to compute the query matrix.
            x_kv: (N, S_kv, E), Tokens from which to compute the key/value matrices.

        Returns:
            (N, S_q, E) + 2 x (N, S_kv, E), query/key/value linear projections of input tokens.
        """
        return self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)


class _QKVMatrixMultiplication(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float, bias: bool = True):
        """Initializes class instance.

        Args:
            d_token: Token size. Must be a multiple of `n_heads`.
            n_heads: Number of attention heads. If greater than 1, then the module will have an additional output layer
                (so called "mixing" layer).
            dropout: Dropout rate for the attention map. The dropout is applied to *probabilities* and does not affect
                logits.
            bias: If `True`, then input (and output, if presented) layers also have bias.
        """
        super().__init__()

        if n_heads > 1:
            if d_token % n_heads != 0:
                raise ValueError("d_token must be a multiple of n_heads")

        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Performs the multiplications between query/key/value matrices.

        Args:
            q: (N, S_q, E), query matrix.
            k: (N, S_kv, E), key matrix.
            v: (N, S_kv, E), value matrix.

        Returns:
            (N, S_q, E), attention output tokens, and attention statistics.
        """
        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, {
            "attention_logits": attention_logits,
            "attention_probs": attention_probs,
        }


class _DifferentiateQKVMatrixMultiplication(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float, depth: int, bias: bool = True):
        """Initializes class instance.

        Args:
            d_token: Token size. Must be a multiple of `n_heads`.
            n_heads: Number of attention heads. If greater than 1, then the module will have an additional output layer
                (so called "mixing" layer).
            dropout: Dropout rate for the attention map. The dropout is applied to *probabilities* and does not affect
                logits.
            depth: Depth of the layer in the transformer stack (used for lambda initialization).
            bias: If `True`, then input (and output, if presented) layers also have bias.
        """
        super().__init__()

        if n_heads > 1:
            if d_token % n_heads != 0:
                raise ValueError("d_token must be a multiple of n_heads")

        self.head_dim = d_token // n_heads // 2
        self.scaling = self.head_dim**-0.5

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        if self.W_out.bias is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape_qk(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape

        return (
            x.reshape(batch_size, n_tokens, self.n_heads * 2, self.head_dim)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads * 2, n_tokens, self.head_dim)
        )

    def _reshape_v(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape

        return (
            x.reshape(batch_size, n_tokens, self.n_heads, self.head_dim * 2)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, self.head_dim * 2)
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Performs the multiplications between query/key/value matrices.

        Args:
            q: (N, S_q, E), query matrix.
            k: (N, S_kv, E), key matrix.
            v: (N, S_kv, E), value matrix.

        Returns:
            (N, S_q, E), attention output tokens, and attention statistics.
        """
        batch_size = len(q)
        n_q_tokens = q.shape[1]
        n_k_tokens = k.shape[1]

        q = self._reshape_qk(q)
        k = self._reshape_qk(k)
        attention_logits = q @ k.transpose(-1, -2) * self.scaling
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attention_probs = attention_probs.view(batch_size * self.n_heads, 2, n_q_tokens, n_k_tokens)
        attention_probs = attention_probs[:, 0] - lambda_full * attention_probs[:, 1]

        x = attention_probs @ self._reshape_v(v)
        x = self.subln(x) * (1 - self.lambda_init)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, self.head_dim * 2)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * self.head_dim * 2)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, {
            "attention_logits": attention_logits,
            "attention_probs": attention_probs,
        }


class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-)."""

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool = True,
        initialization: Literal["kaiming", "xavier"] = "kaiming",
    ) -> None:
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
        self.linear_proj = _QKVLinearProjection(d_token, n_heads, bias=bias, initialization=initialization)
        self.mat_mul = _QKVMatrixMultiplication(d_token, n_heads, dropout, bias=bias)

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Performs a forward pass through the attention operations.

        Args:
            x_q: (N, S_q, E), query tokens.
            x_kv: (N, S_kv, E), key-value tokens.

        Returns:
            (N, S_q, E), attention output tokens, and attention statistics.
        """
        q, k, v = self.linear_proj(x_q, x_kv)
        return self.mat_mul(q, k, v)


class MultiheadDiffAttention(nn.Module):
    """Multihead Attention (self-/cross-)."""

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        dropout: float,
        depth: int,
        bias: bool = False,
        initialization: Literal["kaiming", "xavier"] = "kaiming",
    ) -> None:
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
        self.linear_proj = _QKVLinearProjection(d_token, n_heads, bias=bias, initialization=initialization)
        self.mat_mul = _DifferentiateQKVMatrixMultiplication(d_token, n_heads, dropout, depth, bias=bias)

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Performs a forward pass through the attention operations.

        Args:
            x_q: (N, S_q, E), query tokens.
            x_kv: (N, S_kv, E), key-value tokens.

        Returns:
            (N, S_q, E), attention output tokens, and attention statistics.
        """
        q, k, v = self.linear_proj(x_q, x_kv)
        return self.mat_mul(q, k, v)


class MultiheadCrossAttention(nn.Module):
    """Multihead Cross-Attention."""

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool = True,
        initialization: Literal["kaiming", "xavier"] = "kaiming",
        use_lxmert: bool = False,
    ) -> None:
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
        self.use_lxmert = use_lxmert
        if self.use_lxmert:
            # We follow LXMERT https://github.com/airsplay/lxmert/blob/master/src/lxrt/modeling.py by implementing only one cross-attention
            # module for both tabular and time-series data, to use two times, interverting the inputs in the forward pass
            self.attention_module = MultiheadAttention(d_token, n_heads, dropout, bias, initialization)
        else:
            # Or we can do as follows: building two different attention modules
            self.tabular_attention_module = MultiheadAttention(d_token, n_heads, dropout, bias, initialization)
            self.ts_attention_module = MultiheadAttention(d_token, n_heads, dropout, bias, initialization)

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Performs a forward pass through the attention operations.

        Args:
            x_q: (N, S_q, E), query tokens.
            x_kv: (N, S_kv, E), key-value tokens.

        Returns:
            (N, S_q, E), attention output tokens, and attention statistics.
        """
        if self.use_lxmert:
            # If we follow LXMERT
            tabular_tensor, _ = self.attention_module(x_q, x_kv)
            ts_tensor, _ = self.attention_module(x_kv, x_q)
        else:
            tabular_tensor, _ = self.tabular_attention_module(x_q, x_kv)
            ts_tensor, _ = self.ts_attention_module(x_kv, x_q)

        return tabular_tensor, ts_tensor


class MultiheadDiffCrossAttention(nn.Module):
    """Multihead Cross-Attention."""

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        dropout: float,
        depth: int,
        bias: bool = False,
        initialization: Literal["kaiming", "xavier"] = "kaiming",
        use_lxmert: bool = False,
    ) -> None:
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
        self.use_lxmert = use_lxmert
        if self.use_lxmert:
            # We follow LXMERT https://github.com/airsplay/lxmert/blob/master/src/lxrt/modeling.py by implementing only one cross-attention
            # module for both tabular and time-series data, to use two times, interverting the inputs in the forward pass
            self.attention_module = MultiheadDiffAttention(d_token, n_heads, dropout, depth, bias, initialization)
        else:
            # Or we can do as follows: building two different attention modules
            self.tabular_attention_module = MultiheadDiffAttention(d_token, n_heads, dropout, depth, bias, initialization)
            self.ts_attention_module = MultiheadDiffAttention(d_token, n_heads, dropout, depth, bias, initialization)

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Performs a forward pass through the attention operations.

        Args:
            x_q: (N, S_q, E), query tokens.
            x_kv: (N, S_kv, E), key-value tokens.

        Returns:
            (N, S_q, E), attention output tokens, and attention statistics.
        """
        if self.use_lxmert:
            # If we follow LXMERT
            tabular_tensor, _ = self.attention_module(x_q, x_kv)
            ts_tensor, _ = self.attention_module(x_kv, x_q)
        else:
            tabular_tensor, _ = self.tabular_attention_module(x_q, x_kv)
            ts_tensor, _ = self.ts_attention_module(x_kv, x_q)

        return tabular_tensor, ts_tensor


class PositionalEncoding(nn.Module):
    """Positional encoding layer."""

    def __init__(self, sequence_len: int, d_model: int):
        """Initializes layers parameters.

        Args:
            sequence_len: The number of tokens in the input sequence.
            d_model: The number of features in the input (i.e. the dimensionality of the tokens).
        """
        super().__init__()
        self.positional_encoding = Parameter(torch.empty(sequence_len, d_model))
        init.trunc_normal_(self.positional_encoding, std=0.2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that adds positional encoding to the input tensor.

        Args:
            x: (N, S, `d_model`), Input tensor.

        Returns:
            (N, S, `d_model`), Tensor with added positional encoding.
        """
        return x + self.positional_encoding[None, ...]


class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.

    When used as a module, the [CLS]-token is appended **to the end** of each item in the batch.

    Notes:
        - This is a port of the `CLSToken` class from v0.0.13 of the `rtdl` package. It mixes the original
          implementation with the simpler code of `_CLSEmbedding` from v0.0.2 of the `rtdl_revisiting_models` package.

    References:
        - Original implementation is here: https://github.com/yandex-research/rtdl/blob/f395a2db37bac74f3a209e90511e2cb84e218973/rtdl/modules.py#L380-L446

    Examples:
        .. testcode::

            batch_size = 2
            n_tokens = 3
            d_token = 4
            cls_token = CLSToken(d_token, 'uniform')
            x = torch.randn(batch_size, n_tokens, d_token)
            x = cls_token(x)
            assert x.shape == (batch_size, n_tokens + 1, d_token)
            assert (x[:, -1, :] == cls_token.expand(len(x))).all()
    """

    def __init__(self, d_token: int) -> None:
        """Initializes class instance.

        Args:
            d_token: the size of token
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes the weights using a uniform distribution."""
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def expand(self, *leading_dimensions: int) -> Tensor:
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.

        A possible use case is building a batch of [CLS]-tokens.

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the underlying :code:`weight` parameter, so
            gradients will be propagated as expected.

        Args:
            leading_dimensions: the additional new dimensions

        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Append self **to the end** of each item in the batch (see `CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


class SequencePooling(nn.Module):
    """Sequence pooling layer."""

    def __init__(self, d_model: int):
        """Initializes layer submodules.

        Args:
            d_model: The number of features in the input (i.e. the dimensionality of the tokens).
        """
        super().__init__()
        # Initialize the learnable parameters of the sequential pooling
        self.attention_pool = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that performs a (learnable) weighted averaging of the different tokens.

        Args:
            x: (N, S, `d_model`), Input tensor.

        Returns:
            (N, `d_model`), Output tensor.
        """
        attn_vector = F.softmax(self.attention_pool(x), dim=1)  # (N, S, 1)
        broadcast_attn_vector = attn_vector.transpose(2, 1)  # (N, S, 1) -> (N, 1, S)
        pooled_x = (broadcast_attn_vector @ x).squeeze(1)  # (N, 1, S) @ (N, S, E) -> (N, E)
        return pooled_x


class DownSampling(nn.Module):
    """Downsampling layer for time series data in the sequence dimension."""

    def __init__(self, downsample_factor: int):
        """Initializes class instance.

        Args:
            downsample_factor: Factor by which to downsample the input sequence.
            pooling: Pooling operation to apply. Can be one of ["mean", "max"].
        """
        super().__init__()
        self.downsample_factor = downsample_factor

    def forward(self, x: Tensor) -> Tensor:
        """Downsamples the input tensor along the sequence dimension.

        Args:
            x: (N, S, E), Input tensor to downsample.

        Returns:
            (N, S // `downsample_factor`, E), Downsampled input tensor.
        """
        assert x.ndim == 3, f"Input tensor must have 3 dimensions, but got {x.ndim}."
        assert x.shape[1] % self.downsample_factor == 0, (
            f"Input tensor sequence length must be divisible by the downsample factor, "
            f"but got {x.shape[1]} % {self.downsample_factor} != 0."
        )
        return x[:, :: self.downsample_factor, :]


class LinearPooling(nn.Module):
    """Pooling layer for time series data in the sequence dimension."""

    def __init__(self, n_tokens: int, d_token: int, hidden_dim: Optional[int] = None, transpose: bool = False):
        """Initializes class instance.

        Args:
            n_tokens: Number of tokens in the input tensor
        """
        super().__init__()
        if hidden_dim is not None:
            self.linear_pool = nn.Sequential(nn.Linear(n_tokens, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_token))
        else:
            self.linear_pool = nn.Sequential(nn.Linear(n_tokens, d_token))

        self.transpose = transpose

    def forward(self, x: Tensor) -> Tensor:
        """Applies a linear projection along the sequence dimension of the input tensor.

        Args:
            x: (N, S, E), Input tensor to pool.

        Returns:
            (N, S_out, E) or (N, S, E_out), Output tensor.
        """
        if self.transpose:
            return self.linear_pool(x.transpose(1, 2)).transpose(1, 2)
        else:
            return self.linear_pool(x)


class TS_Patching(nn.Module):
    """Downsampling layer for time series data."""

    def __init__(
        self, in_features: int, out_features: int, kernel_size: int, stride: int, padding=0, transpose: bool = False
    ):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input tensor.
            out_features: Number of features in the output tensor.
            kernel_size: Size of the sliding window.
            stride: Stride of the sliding window.
        """
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding)
        self.transpose = transpose

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the downsampling layer.

        Args:
            x: (N, S_ts_raw, E), Input tensor.

        Returns:
            (N, S_ts, E), Output tensor.
        """
        if self.transpose:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)
        else:
            return self.conv(x)


class MultiResolutionPatching(nn.Module):
    """Multi-resolution patching layer for time series data."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_sizes: List[int],
        strides: List[int],
        padding: List[int],
    ):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input tensor.
            out_features: Number of features in the output tensor.
            kernel_sizes: Sizes of the sliding windows.
            strides: Strides of the sliding windows.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_features, out_features, kernel_size=kernel_sizes[0], stride=strides[0], padding=padding[0]
        )
        self.conv2 = nn.Conv1d(
            in_features, out_features, kernel_size=kernel_sizes[1], stride=strides[1], padding=padding[1]
        )

        # self.conv3 = nn.Conv1d(
        #     in_features, out_features, kernel_size=kernel_sizes[2], stride=strides[2], padding=padding[2]
        # )

    def forward(self, x: List[Tensor]) -> Tensor:
        """Performs a forward pass through the multi-resolution patching layer.

        Args:
            x: (N, num_ts, S_ts_raw, 1), Input tensor.

        Returns:
            (N, S_ts, 1, Output tensor.
        """
        # return torch.cat(
        #     [self.conv1(x.transpose(1, 2)).transpose(1, 2), self.conv2(x.transpose(1, 2)).transpose(1, 2)], dim=1
        # )
        conv_results = torch.cat(
            [
                self.conv1(x[0].unsqueeze(-1).transpose(1, 2)).transpose(1, 2),
                self.conv2(x[1].unsqueeze(-1).transpose(1, 2)).transpose(1, 2),
                # self.conv3(x[2].unsqueeze(-1).transpose(1, 2)).transpose(1, 2),
            ],
            dim=1,
        )  # (N, S_ts, 1)
        return conv_results


class FTPredictionHead(nn.Module):
    """Prediction head architecture described in the Feature Tokenizer transformer (FT-Transformer) paper."""

    def __init__(self, in_features: int, out_features: int):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input feature vector.
            out_features: Number of features to output.
        """
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(in_features), nn.ReLU(), nn.Linear(in_features, out_features))

    def forward(self, x: Tensor) -> Tensor:
        """Predicts unnormalized features from a feature vector input.

        Args:
            x: (N, `in_features`), Batch of feature vectors.

        Returns:
            - (N, `out_features`), Batch of output features.
        """
        # if type(x) == tuple:
        #     x = x[0]

        return self.head(x)


class TabPFNPredictionHead(nn.Module):
    """Prediction head architecture described in the TabPFN paper."""

    def __init__(
        self, in_features: int, out_features: int, use_layernorm: bool = False, model_path: Optional[str] = None, updated_pfn_path: Optional[str] = None
    ):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input feature vector.
            out_features: Number of features to output.
        """
        super().__init__()
        self.model_path = model_path
        self.n_class = out_features

        if self.model_path is None:
            module_dict = nn.ModuleDict()
            if use_layernorm:
                module_dict["layernorm"] = nn.LayerNorm(in_features)
            module_dict["fc1"] = nn.Linear(in_features, 4 * in_features)
            module_dict["gelu"] = nn.GELU()
            module_dict["fc2"] = nn.Linear(4 * in_features, out_features)

            self.head = nn.Sequential(*module_dict.values())

        else:
            # self.head = load_model_criterion_config(
            #     model_path=model_path,
            #     check_bar_distribution_criterion=False,
            #     cache_trainset_representation=False,
            #     which="classifier",
            #     version="v2",
            #     download_if_not_exists=False,
            # )[0][0].decoder_dict["standard"]
            
            list_model, _, _, _ = load_model_criterion_config(
                model_path=model_path,
                check_bar_distribution_criterion=False,
                cache_trainset_representation=False,
                which="classifier",
                version="v2",
                download_if_not_exists=False,
            )
            model = list_model[0]
            
            if updated_pfn_path is not None:
                # Load updated model weights after pretraining
                state_dict = torch.load(updated_pfn_path, map_location="cuda:0")
                # new_state_dict = {}
                # for k, v in state_dict.items():
                #     if k.startswith("model."):
                #         new_key = k[len("model.") :]  # strip the prefix
                #         new_state_dict[new_key] = v
                #     else:
                #         new_state_dict[k] = v
                if "pe" in state_dict:
                    state_dict.pop("pe")  # Remove pe if present
                if "channel_positional_encoding" in state_dict:
                    state_dict.pop("channel_positional_encoding")  # Remove channel_positional_encoding if present
                model.load_state_dict(state_dict, strict=True)

            self.head = model.decoder_dict["standard"]

    def forward(self, x: Tensor) -> Tensor:
        """Predicts unnormalized features from a feature vector input.

        Args:
            x: (N, `in_features`), Batch of feature vectors.

        Returns:
            - (N, `out_features`), Batch of output features.
        """
        if self.model_path is None:
            x = self.head(x)
        else:
            x = self.head(x.unsqueeze(1)).squeeze(1)
            x = x[:, : self.n_class]

        return x


class UnimodalLogitsHead(nn.Module):
    """Layer to output (enforced) unimodal logits from an input feature vector.

    This is a re-implementation of a 2017 ICML paper by Beckham and Pal, which proposes to use either a Poisson or
    binomial distribution to output unimodal logits (because they are constrained as such by the distribution) from a
    scalar value.

    References:
        - ICML 2017 paper: https://proceedings.mlr.press/v70/beckham17a.html
    """

    def __init__(
        self,
        in_features: int,
        num_logits: int,
        distribution: Literal["poisson", "binomial"] = "binomial",
        tau: float = 1.0,
        tau_mode: Literal["fixed", "learn", "learn_sigm", "learn_fn"] = "learn_fn",
        eps: float = 1e-6,
    ):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input feature vector.
            num_logits: Number of logits to output.
            distribution: Distribution whose probability mass function (PMF) is used to enforce an unimodal distribution
                of the logits.
            tau: Temperature parameter to control the sharpness of the distribution.
                - If `tau_mode` is 'fixed', this is the fixed value of tau.
                - If `tau_mode` is 'learn' or 'learn_sigm', this is the initial value of tau.
                - If `tau_mode` is 'learn_fn', this argument is ignored.
            tau_mode: Method to use to set or learn the temperature parameter:
                - 'fixed': Use a fixed value of tau.
                - 'learn': Learn tau.
                - 'learn_sigm': Learn tau through a sigmoid function.
                - 'learn_fn': Learn tau through a function of the input, i.e. a tau that varies for each input.
                  The function is 1 / (1 + g(L(x))), where g is the softplus function. and L is a linear layer.
            eps: Epsilon value to use in probabilities' log to avoid numerical instability.
        """
        super().__init__()
        self.num_logits = num_logits
        self.distribution = distribution
        self.tau_mode = tau_mode
        self.eps = eps

        self.register_buffer("logits_idx", torch.arange(self.num_logits))
        match self.distribution:
            case "poisson":
                self.register_buffer("logits_factorial", torch.from_numpy(factorial(self.logits_idx)))
            case "binomial":
                self.register_buffer("binom_coef", binom(self.num_logits - 1, self.logits_idx))
            case _:
                raise ValueError(f"Unsupported distribution '{distribution}'.")

        self.param_head = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())

        match self.tau_mode:
            case "fixed":
                self.tau = torch.tensor(float(tau))
            case "learn" | "learn_sigm":
                self.tau = nn.Parameter(torch.tensor(float(tau)))
            case "learn_fn":
                self.tau_head = nn.Sequential(nn.Linear(in_features, 1), nn.Softplus())
            case _:
                raise ValueError(f"Unsupported tau mode '{tau_mode}'.")

    def __repr__(self):
        """Overrides the default repr to display the important parameters of the layer."""
        vars = {"in_features": self.param_head[0].in_features}
        vars.update({var: getattr(self, var) for var in ["num_logits", "distribution", "tau_mode"]})
        vars_str = [f"{var}={val}" for var, val in vars.items()]
        return f"{self.__class__.__name__}({', '.join(vars_str)})"

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Predicts unnormalized, unimodal logits from a feature vector input.

        Args:
            x: (N, `in_features`), Batch of feature vectors.

        Returns:
            - (N, `num_logits`), Output tensor of unimodal logits. The logits are unnormalized, but the temperature
              to control the sharpness of the distribution as already been applied.
            - (N, 1), Predicted parameter of the distribution, in the range [0, 1].
            - (N, 1), Temperature parameter tau, in the range [0, inf) or [0, 1], depending on `tau_mode`.
        """
        # Forward through the linear layer to get a scalar param in [0,1] for the distribution
        f_x = param = self.param_head(x)

        # Compute the probability mass function (PMF) of the distribution
        # For technical reasons, use the log instead of the direct value
        match self.distribution:
            case "poisson":
                f_x = (self.num_logits + 1) * f_x  # Rescale f(x) to [0, num_logits+1]
                log_f = (self.logits_idx * torch.log(f_x + self.eps)) - f_x - torch.log(self.logits_factorial)
            case "binomial":
                log_f = (
                    torch.log(self.binom_coef)
                    + (self.logits_idx * torch.log(f_x + self.eps))
                    + ((self.num_logits - 1 - self.logits_idx) * torch.log(1 - f_x + self.eps))
                )

        # Compute the temperature parameter tau
        # In cases where tau is a scalar, manually broadcast it to a tensor w/ one value for each item in the batch
        # This is done to keep a consistent API for the different tau modes, with tau having a different value for each
        # item in the batch when `tau_mode` is 'learn_fn'
        match self.tau_mode:
            case "fixed":
                tau = torch.full_like(param, self.tau.item())  # Manual broadcast
            case "learn":
                tau = self.tau.expand_as(param)  # Manual broadcast
            case "learn_sigm":
                tau = torch.sigmoid(self.tau).expand_as(param)  # Sigmoid + manual broadcast
            case "learn_fn":
                tau = 1 / (1 + self.tau_head(x))
            case _:
                raise ValueError(f"Unsupported 'tau_mode': '{self.tau_mode}'.")

        return log_f / tau, param, tau
