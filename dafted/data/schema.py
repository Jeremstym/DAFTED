"""Dataset schema: the single place where `dafted` learns which attributes are available in the data.

`dafted` does not hard-code the variables of any dataset. The tabular attributes (numerical or categorical), the
time-series attributes and the views available in the data are read from a YAML schema file, whose path is given by the
`DAFTED_SCHEMA` environment variable. When it is not set, the schema of the synthetic dataset bundled with the package
is used (see `dafted/data/schemas/synthetic.yaml`, which also documents the format of the file).

The enums built from the schema (`TabularAttribute`, `TimeSeriesAttribute`, `View`) are string enums, so they compare
equal to (and hash like) plain strings or members of any other string enum with the same values. This means a custom
data module can use plain strings, or its own enums, as keys of the items it yields.
"""

import os
from enum import auto
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, TypeVar, Union

import numpy as np
import yaml
from strenum import SnakeCaseStrEnum, StrEnum
from torch import Tensor

from dafted.core.data.config import Subset

SCHEMA_ENV_VAR = "DAFTED_SCHEMA"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "synthetic.yaml"

MISSING_NUM_ATTR = np.nan
"""Value flagging a missing numerical attribute."""
MISSING_CAT_ATTR = -1
"""Value flagging a missing categorical attribute (otherwise encoded by the index of its label)."""

PREDICT_DATALOADERS_SUBSETS = [Subset.TRAIN, Subset.VAL, Subset.TEST]
"""Subsets for which data modules must provide a dataloader (in this order) in `predict_dataloader`."""

NUMERICAL = "numerical"
"""Keyword marking a tabular attribute as numerical in the schema file."""


def get_schema_path() -> Path:
    """Resolves the path of the schema file describing the dataset.

    Returns:
        Path of the schema file, read from the `DAFTED_SCHEMA` environment variable, or the bundled synthetic schema.
    """
    return Path(os.environ.get(SCHEMA_ENV_VAR) or DEFAULT_SCHEMA_PATH).expanduser()


def load_schema(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads and validates a dataset schema file.

    Args:
        path: Path of the YAML schema file.

    Returns:
        Schema, with the keys `views`, `time_series_attrs`, `tabular_attrs` and `ordinal_attrs`.
    """
    schema = yaml.safe_load(Path(path).read_text()) or {}
    schema.setdefault("views", [])
    schema.setdefault("time_series_attrs", [])
    schema.setdefault("ordinal_attrs", [])
    if not schema.get("tabular_attrs"):
        raise ValueError(f"Schema '{path}' must declare at least one tabular attribute under `tabular_attrs`.")

    for attr, attr_type in schema["tabular_attrs"].items():
        if attr_type != NUMERICAL and not (isinstance(attr_type, list) and len(attr_type) >= 2):
            raise ValueError(
                f"Invalid type for tabular attribute '{attr}' in schema '{path}': '{attr_type}'. It should be either "
                f"'{NUMERICAL}' or the list of the labels (at least 2) of a categorical attribute."
            )
    if unknown_ordinal_attrs := [
        attr for attr in schema["ordinal_attrs"] if not isinstance(schema["tabular_attrs"].get(attr), list)
    ]:
        raise ValueError(f"Ordinal attributes {unknown_ordinal_attrs} are not categorical attributes in '{path}'.")
    return schema


_SCHEMA_PATH = get_schema_path()
_SCHEMA = load_schema(_SCHEMA_PATH)


def _members(names: Sequence[str]) -> List[Tuple[str, str]]:
    # Assign values explicitly, so that they are identical to the names in the schema (e.g. even if they contain digits)
    return [(str(name), str(name)) for name in names]


class _TabularAttributeBase(StrEnum):
    """Base class for `TabularAttribute`, providing the queries on the types of the attributes."""

    @classmethod
    def categorical_attrs(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that are categorical."""
        return [attr for attr in cls if attr in TABULAR_CAT_ATTR_LABELS]

    @classmethod
    def numerical_attrs(cls) -> List["TabularAttribute"]:
        """Lists the tabular attributes that are numerical/continuous."""
        return [attr for attr in cls if attr not in TABULAR_CAT_ATTR_LABELS]

    @classmethod
    def ordinal_attrs(cls) -> List["TabularAttribute"]:
        """Lists the subset of categorical attributes that are ordinal (i.e. the ordering of classes is meaningful)."""
        return [cls[attr] for attr in _SCHEMA["ordinal_attrs"]]

    @classmethod
    def binary_attrs(cls) -> List["TabularAttribute"]:
        """Lists the subset of categorical attributes that only have 2 classes (e.g. bool)."""
        return [attr for attr in cls.categorical_attrs() if len(TABULAR_CAT_ATTR_LABELS[attr]) == 2]

    @classmethod
    def boolean_attrs(cls) -> List["TabularAttribute"]:
        """Lists the subset of binary attributes that are boolean."""
        return [attr for attr in cls.binary_attrs() if TABULAR_CAT_ATTR_LABELS[attr] == [False, True]]


TabularAttribute = _TabularAttributeBase(
    "TabularAttribute", _members(_SCHEMA["tabular_attrs"]), module=__name__, qualname="TabularAttribute"
)
"""Scalar attributes of a patient (e.g. from the patient's record), declared under `tabular_attrs` in the schema."""

TimeSeriesAttribute = StrEnum(
    "TimeSeriesAttribute", _members(_SCHEMA["time_series_attrs"]), module=__name__, qualname="TimeSeriesAttribute"
)
"""Attributes that are temporal sequences of values, declared under `time_series_attrs` in the schema."""

View = StrEnum("View", _members(_SCHEMA["views"]), module=__name__, qualname="View")
"""Views (e.g. acquisition views) for which time-series attributes are available, declared under `views`."""

TABULAR_CAT_ATTR_LABELS: Dict[TabularAttribute, List[Any]] = {
    TabularAttribute[attr]: labels for attr, labels in _SCHEMA["tabular_attrs"].items() if labels != NUMERICAL
}
"""Labels of the categorical tabular attributes. Categorical values are encoded by the index of their label."""


class DataTag(SnakeCaseStrEnum):
    """Tags referring to the different modalities of data, used e.g. as keys of `DataParameters.in_shape`."""

    time_series_attrs = auto()
    tabular_attrs = auto()


T = TypeVar("T", np.ndarray, Tensor)
PatientData = Dict[Union[str, TabularAttribute, View], Union[str, T, Dict[TimeSeriesAttribute, T]]]
"""Item (or batch) yielded by data modules, i.e. `{"id": ..., <tabular attr>: value, <view>: {<ts attr>: values}}`."""


def filter_time_series_attributes(
    item_or_batch: PatientData,
    views: Sequence[View] = tuple(View),
    attrs: Sequence[TimeSeriesAttribute] = tuple(TimeSeriesAttribute),
) -> Dict[Tuple[View, TimeSeriesAttribute], T]:
    """Filters an item/batch of data to only keep the requested time-series attributes from some views.

    Args:
        item_or_batch: Item/batch of data from a data module.
        views: Views for which to keep time-series attributes data.
        attrs: Attributes for which to keep data.

    Returns:
        Requested time-series attributes from the requested views in the item/batch of data.
    """
    return {
        (view, view_data_tag): data
        for view in views
        for view_data_tag, data in item_or_batch.get(view, {}).items()
        if view_data_tag in attrs
    }
