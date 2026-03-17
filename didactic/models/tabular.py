import json
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from dataprocessing.data.orchid.config import OrchidTag, TabularAttribute, TimeSeriesAttribute
from dataprocessing.data.orchid.datapipes import MISSING_CAT_ATTR
from rtdl_revisiting_models import CategoricalEmbeddings, LinearEmbeddings  # type: ignore[import-untyped]
from torch import Tensor, nn


def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


class TabularEmbedding(nn.Module):
    """Combines `LinearEmbeddings` and `CategoricalEmbeddings`.

    The "Feature Tokenizer" module from "Revisiting Deep Learning Models for Tabular Data" by Gorishniy et al. (2021).
    The module transforms continuous and categorical features to tokens (embeddings).

    Notes:
        - This is a port of the `FeatureTokenizer` class from v0.0.13 of the `rtdl` package using the updated underlying
          `CategoricalEmbeddings` and `LinearEmbeddings` from v0.0.2 of the `rtdl_revisiting_models` package, instead of
          the original `CategoricalFeatureTokenizer` and `NumericalFeatureTokenizer` from the `rtdl` package.

    References:
        - Original implementation is here: https://github.com/yandex-research/rtdl/blob/f395a2db37bac74f3a209e90511e2cb84e218973/rtdl/modules.py#L260-L377

    Examples:
        .. testcode::

            n_objects = 4
            n_num_features = 3
            n_cat_features = 2
            d_token = 7
            x_num = torch.randn(n_objects, n_num_features)
            x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
            # [2, 3] reflects cardinalities
            tokenizer = FeatureTokenizer(n_num_features, [2, 3], d_token)
            tokens = tokenizer(x_num, x_cat)
            assert tokens.shape == (n_objects, n_num_features + n_cat_features, d_token)
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
    ) -> None:
        """Initializes class instance.

        Args:
            n_num_features: the number of continuous features. Pass :code:`0` if there are no numerical features.
            cat_cardinalities: the number of unique values for each feature. Pass an empty list if there are no
                categorical features.
            d_token: the size of one token.
        """
        super().__init__()
        assert n_num_features >= 0, "n_num_features must be non-negative"
        assert (
            n_num_features or cat_cardinalities
        ), "at least one of n_num_features or cat_cardinalities must be positive/non-empty"
        self.num_tokenizer = LinearEmbeddings(n_num_features, d_token) if n_num_features else None
        self.cat_tokenizer = CategoricalEmbeddings(cat_cardinalities, d_token) if cat_cardinalities else None

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return sum(x.n_tokens for x in [self.num_tokenizer, self.cat_tokenizer] if x is not None)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.cat_tokenizer.d_token if self.num_tokenizer is None else self.num_tokenizer.d_token  # type: ignore

    # def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
    def forward(
        self,
        tabular_attrs: Dict[TabularAttribute, Tensor],
        tabular_num_attrs: List[TabularAttribute],
        tabular_cat_attrs: List[TabularAttribute],
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Perform the forward pass.

        Args:
            #DEPRECATED x_num: continuous features. Must be presented if :code:`n_num_features > 0` was passed to the constructor.
            #DEPRECATED x_cat: categorical features. Must be presented if non-empty :code:`cat_cardinalities` was passed to the
                constructor.
            tabular_attrs: (K: S, V: N), Sequence of batches of tabular attributes. To indicate an item is missing an
                attribute, the flags `MISSING_NUM_ATTR`/`MISSING_CAT_ATTR` can be used for numerical and categorical
                attributes, respectively.
            tabular_num_attrs: boolean flag indicating whether numerical attributes are present in the `tabular_attrs`
                input.
            tabular_cat_attrs: boolean flag indicating whether categorical attributes are present in the `tabular_attrs`
                input.

        Returns:
            tokens

        Raises:
            AssertionError: if the described requirements for the inputs are not met.
        """
        # assert x_num is not None or x_cat is not None, "At least one of x_num and x_cat must be presented"
        num_attrs, cat_attrs = None, None
        if tabular_num_attrs:
            # Group the numerical attributes from the `tabular_attrs` input in a single tensor
            num_attrs = torch.hstack([tabular_attrs[attr].unsqueeze(1) for attr in tabular_num_attrs])  # (N, S_num)
        if tabular_cat_attrs:
            # Group the categorical attributes from the `tabular_attrs` input in a single tensor
            cat_attrs = torch.hstack([tabular_attrs[attr].unsqueeze(1) for attr in tabular_cat_attrs])  # (N, S_cat)

        x_num = torch.nan_to_num(num_attrs) if num_attrs is not None else None
        x_cat = cat_attrs.clip(0) if cat_attrs is not None else None
        assert _all_or_none(
            [self.num_tokenizer, x_num]
        ), "If self.num_tokenizer is (not) None, then x_num must (not) be None"
        assert _all_or_none(
            [self.cat_tokenizer, x_cat]
        ), "If self.cat_tokenizer is (not) None, then x_cat must (not) be None"
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))

        tab_notna_mask: Union[Tensor, Tuple[Tensor, Tensor]]
        if tabular_num_attrs and tabular_cat_attrs:
            assert num_attrs is not None and cat_attrs is not None
            tab_num_notna_mask = ~(num_attrs.isnan())
            tab_cat_notna_mask = cat_attrs != MISSING_CAT_ATTR
            tab_notna_mask = (tab_num_notna_mask, tab_cat_notna_mask)
        elif tabular_num_attrs:
            assert num_attrs is not None
            tab_num_notna_mask = ~(num_attrs.isnan())
            tab_notna_mask = tab_num_notna_mask
        elif tabular_cat_attrs:
            assert cat_attrs is not None
            tab_cat_notna_mask = cat_attrs != MISSING_CAT_ATTR
            tab_notna_mask = tab_cat_notna_mask
        else:
            raise ValueError("At least one of tabular_num_attrs or tabular_cat_attrs must be non-empty")

        return (x[0], tab_notna_mask) if len(x) == 1 else (torch.cat(x, dim=1), tab_notna_mask)


class TabularIdentityEmbedding(nn.Module):
    """Combines `LinearEmbeddings` and `CategoricalEmbeddings`.

    The "Feature Tokenizer" module from "Revisiting Deep Learning Models for Tabular Data" by Gorishniy et al. (2021).
    The module transforms continuous and categorical features to tokens (embeddings).

    Notes:
        - This is a port of the `FeatureTokenizer` class from v0.0.13 of the `rtdl` package using the updated underlying
          `CategoricalEmbeddings` and `LinearEmbeddings` from v0.0.2 of the `rtdl_revisiting_models` package, instead of
          the original `CategoricalFeatureTokenizer` and `NumericalFeatureTokenizer` from the `rtdl` package.

    References:
        - Original implementation is here: https://github.com/yandex-research/rtdl/blob/f395a2db37bac74f3a209e90511e2cb84e218973/rtdl/modules.py#L260-L377

    Examples:
        .. testcode::

            n_objects = 4
            n_num_features = 3
            n_cat_features = 2
            d_token = 7
            x_num = torch.randn(n_objects, n_num_features)
            x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
            # [2, 3] reflects cardinalities
            tokenizer = FeatureTokenizer(n_num_features, [2, 3], d_token)
            tokens = tokenizer(x_num, x_cat)
            assert tokens.shape == (n_objects, n_num_features + n_cat_features, d_token)
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
    ) -> None:
        """Initializes class instance.

        Args:
            n_num_features: the number of continuous features. Pass :code:`0` if there are no numerical features.
            cat_cardinalities: the number of unique values for each feature. Pass an empty list if there are no
                categorical features.
            d_token: the size of one token.
        """
        super().__init__()
        assert n_num_features >= 0, "n_num_features must be non-negative"
        assert (
            n_num_features or cat_cardinalities
        ), "at least one of n_num_features or cat_cardinalities must be positive/non-empty"
        self.num_tokenizer = LinearEmbeddings(n_num_features, d_token) if n_num_features else None
        self.cat_tokenizer = CategoricalEmbeddings(cat_cardinalities, d_token) if cat_cardinalities else None

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return sum(x.n_tokens for x in [self.num_tokenizer, self.cat_tokenizer] if x is not None)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.cat_tokenizer.d_token if self.num_tokenizer is None else self.num_tokenizer.d_token  # type: ignore

    # def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
    def forward(
        self,
        tabular_attrs: Dict[TabularAttribute, Tensor],
        tabular_num_attrs: List[TabularAttribute],
        tabular_cat_attrs: List[TabularAttribute],
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Perform the forward pass.

        Args:
            #DEPRECATED x_num: continuous features. Must be presented if :code:`n_num_features > 0` was passed to the constructor.
            #DEPRECATED x_cat: categorical features. Must be presented if non-empty :code:`cat_cardinalities` was passed to the
                constructor.
            tabular_attrs: (K: S, V: N), Sequence of batches of tabular attributes. To indicate an item is missing an
                attribute, the flags `MISSING_NUM_ATTR`/`MISSING_CAT_ATTR` can be used for numerical and categorical
                attributes, respectively.
            tabular_num_attrs: List of numerical attributes to be used in the embedding.
            tabular_cat_attrs: List of categorical attributes to be used in the embedding.

        Returns:
            tokens

        Raises:
            AssertionError: if the described requirements for the inputs are not met.
        """
        # assert x_num is not None or x_cat is not None, "At least one of x_num and x_cat must be presented"
        num_attrs, cat_attrs = None, None
        x = []
        if tabular_num_attrs:
            # Group the numerical attributes from the `tabular_attrs` input in a single tensor
            num_attrs = torch.hstack([tabular_attrs[attr].unsqueeze(1) for attr in tabular_num_attrs])  # (N, S_num)
        if tabular_cat_attrs:
            # Group the categorical attributes from the `tabular_attrs` input in a single tensor
            cat_attrs = torch.hstack([tabular_attrs[attr].unsqueeze(1) for attr in tabular_cat_attrs])  # (N, S_cat)

        num_attrs = torch.nan_to_num(num_attrs) if num_attrs is not None else None
        x.append(num_attrs) if num_attrs is not None else None
        cat_attrs = cat_attrs.clip(0) if cat_attrs is not None else None
        x.append(cat_attrs) if cat_attrs is not None else None
        tab_attrs_tokens = torch.cat(x, dim=1)  # (N, S_tab)
        tab_attrs_tokens = tab_attrs_tokens.unsqueeze(-1)  # (N, S_tab, 1)
        tab_attrs_tokens = tab_attrs_tokens.repeat(1, 1, self.hparams.embed_dim)  # (N, S_tab, E)

        tab_notna_mask: Union[Tensor, Tuple[Tensor, Tensor]]
        if tabular_num_attrs and tabular_cat_attrs:
            assert num_attrs is not None and cat_attrs is not None
            tab_num_notna_mask = ~(num_attrs.isnan())
            tab_cat_notna_mask = cat_attrs != MISSING_CAT_ATTR
            tab_notna_mask = (tab_num_notna_mask, tab_cat_notna_mask)
        elif tabular_num_attrs:
            assert num_attrs is not None
            tab_num_notna_mask = ~(num_attrs.isnan())
            tab_notna_mask = tab_num_notna_mask
        elif tabular_cat_attrs:
            assert cat_attrs is not None
            tab_cat_notna_mask = cat_attrs != MISSING_CAT_ATTR
            tab_notna_mask = tab_cat_notna_mask
        else:
            raise ValueError("At least one of tabular_num_attrs or tabular_cat_attrs must be non-empty")

        return tab_attrs_tokens, tab_notna_mask


class TabularLinearEmbedding(nn.Module):
    """Combines `LinearEmbeddings` and `CategoricalEmbeddings`.

    The "Feature Tokenizer" module from "Revisiting Deep Learning Models for Tabular Data" by Gorishniy et al. (2021).
    The module transforms continuous and categorical features to tokens (embeddings).

    Notes:
        - This is a port of the `FeatureTokenizer` class from v0.0.13 of the `rtdl` package using the updated underlying
          `CategoricalEmbeddings` and `LinearEmbeddings` from v0.0.2 of the `rtdl_revisiting_models` package, instead of
          the original `CategoricalFeatureTokenizer` and `NumericalFeatureTokenizer` from the `rtdl` package.

    References:
        - Original implementation is here: https://github.com/yandex-research/rtdl/blob/f395a2db37bac74f3a209e90511e2cb84e218973/rtdl/modules.py#L260-L377

    Examples:
        .. testcode::

            n_objects = 4
            n_num_features = 3
            n_cat_features = 2
            d_token = 7
            x_num = torch.randn(n_objects, n_num_features)
            x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
            # [2, 3] reflects cardinalities
            tokenizer = FeatureTokenizer(n_num_features, [2, 3], d_token)
            tokens = tokenizer(x_num, x_cat)
            assert tokens.shape == (n_objects, n_num_features + n_cat_features, d_token)
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
    ) -> None:
        """Initializes class instance.

        Args:
            n_num_features: the number of continuous features. Pass :code:`0` if there are no numerical features.
            cat_cardinalities: the number of unique values for each feature. Pass an empty list if there are no
                categorical features.
            d_token: the size of one token.
        """
        super().__init__()
        assert n_num_features >= 0, "n_num_features must be non-negative"
        assert (
            n_num_features or cat_cardinalities
        ), "at least one of n_num_features or cat_cardinalities must be positive/non-empty"
        self.num_tokenizer = LinearEmbeddings(n_num_features, d_token) if n_num_features else None
        self.cat_tokenizer = LinearEmbeddings(len(cat_cardinalities), d_token) if cat_cardinalities else None

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return sum(x.n_tokens for x in [self.num_tokenizer, self.cat_tokenizer] if x is not None)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.cat_tokenizer.d_token if self.num_tokenizer is None else self.num_tokenizer.d_token  # type: ignore

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        """Perform the forward pass.

        Args:
            x_num: continuous features. Must be presented if :code:`n_num_features > 0` was passed to the constructor.
            x_cat: categorical features. Must be presented if non-empty :code:`cat_cardinalities` was passed to the
                constructor.

        Returns:
            tokens

        Raises:
            AssertionError: if the described requirements for the inputs are not met.
        """

        assert x_num is not None or x_cat is not None, "At least one of x_num and x_cat must be presented"
        assert _all_or_none(
            [self.num_tokenizer, x_num]
        ), "If self.num_tokenizer is (not) None, then x_num must (not) be None"
        assert _all_or_none(
            [self.cat_tokenizer, x_cat]
        ), "If self.cat_tokenizer is (not) None, then x_cat must (not) be None"
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)


class TabularOrdinalEmbedding(nn.Module):
    """Combines `LinearEmbeddings` and `CategoricalEmbeddings`.

    The "Feature Tokenizer" module from "Revisiting Deep Learning Models for Tabular Data" by Gorishniy et al. (2021).
    The module transforms continuous and categorical features to tokens (embeddings).

    Notes:
        - This is a port of the `FeatureTokenizer` class from v0.0.13 of the `rtdl` package using the updated underlying
          `CategoricalEmbeddings` and `LinearEmbeddings` from v0.0.2 of the `rtdl_revisiting_models` package, instead of
          the original `CategoricalFeatureTokenizer` and `NumericalFeatureTokenizer` from the `rtdl` package.

    References:
        - Original implementation is here: https://github.com/yandex-research/rtdl/blob/f395a2db37bac74f3a209e90511e2cb84e218973/rtdl/modules.py#L260-L377

    Examples:
        .. testcode::

            n_objects = 4
            n_num_features = 3
            n_cat_features = 2
            d_token = 7
            x_num = torch.randn(n_objects, n_num_features)
            x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
            # [2, 3] reflects cardinalities
            tokenizer = FeatureTokenizer(n_num_features, [2, 3], d_token)
            tokens = tokenizer(x_num, x_cat)
            assert tokens.shape == (n_objects, n_num_features + n_cat_features, d_token)
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
    ) -> None:
        """Initializes class instance.

        Args:
            n_num_features: the number of continuous features. Pass :code:`0` if there are no numerical features.
            cat_cardinalities: the number of unique values for each feature. Pass an empty list if there are no
                categorical features.
            d_token: the size of one token.
        """
        super().__init__()
        assert n_num_features >= 0, "n_num_features must be non-negative"
        assert (
            n_num_features or cat_cardinalities
        ), "at least one of n_num_features or cat_cardinalities must be positive/non-empty"
        self.num_tokenizer = LinearEmbeddings(n_num_features, d_token) if n_num_features else None
        self.cat_tokenizer = CategoricalEmbeddings(cat_cardinalities, d_token) if cat_cardinalities else None

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return sum(x.n_tokens for x in [self.num_tokenizer, self.cat_tokenizer] if x is not None)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.cat_tokenizer.d_token if self.num_tokenizer is None else self.num_tokenizer.d_token  # type: ignore

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        """Perform the forward pass.

        Args:
            x_num: continuous features. Must be presented if :code:`n_num_features > 0` was passed to the constructor.
            x_cat: categorical features. Must be presented if non-empty :code:`cat_cardinalities` was passed to the
                constructor.

        Returns:
            tokens

        Raises:
            AssertionError: if the described requirements for the inputs are not met.
        """

        assert x_num is not None or x_cat is not None, "At least one of x_num and x_cat must be presented"
        assert _all_or_none(
            [self.num_tokenizer, x_num]
        ), "If self.num_tokenizer is (not) None, then x_num must (not) be None"
        assert _all_or_none(
            [self.cat_tokenizer, x_cat]
        ), "If self.cat_tokenizer is (not) None, then x_cat must (not) be None"
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)
