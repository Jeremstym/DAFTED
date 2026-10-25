"""Synthetic dataset, to run DAFTED end-to-end without access to private data, and template for custom data modules.

Any data module can be used with DAFTED, as long as it:
    - inherits from `dafted.core.data.data_module.VitalDataModule`;
    - yields items in the `dafted.data.schema.PatientData` format, i.e.
      `{"id": str, <tabular attr>: 0-d array, <view>: {<time-series attr>: 1-d array}}`, where numerical attributes are
      `float32` (`MISSING_NUM_ATTR` if missing) and categorical attributes are the `int64` index of their label in the
      schema (`MISSING_CAT_ATTR` if missing);
    - provides `data_params.in_shape` with the shapes of the modalities, under the `DataTag` keys;
    - returns one unbatched dataloader per subset in `PREDICT_DATALOADERS_SUBSETS` from `predict_dataloader`, and
      exposes the patients of each subset (in the same order) in `subsets_patients`, as a mapping from patient IDs to
      objects with `id` and `attrs` (raw values of the tabular attributes) fields, like `PatientRecord`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from pytorch_lightning.trainer.states import TrainerFn
from scipy import stats
from torch.utils.data import DataLoader, Dataset

from dafted.core.data.config import DataParameters, Subset
from dafted.core.data.data_module import VitalDataModule
from dafted.data.schema import (
    MISSING_CAT_ATTR,
    MISSING_NUM_ATTR,
    PREDICT_DATALOADERS_SUBSETS,
    TABULAR_CAT_ATTR_LABELS,
    DataTag,
    PatientData,
    TabularAttribute,
    TimeSeriesAttribute,
    View,
)


@dataclass
class PatientRecord:
    """Raw data of a patient, i.e. with the labels of categorical attributes instead of their encoded indices."""

    id: str
    attrs: Dict[TabularAttribute, Any] = field(default_factory=dict)
    """Values of the patient's available tabular attributes (missing attributes are left out)."""


class _ItemsDataset(Dataset):
    """Map-style dataset over a list of pre-generated items."""

    def __init__(self, items: List[PatientData]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> PatientData:
        return self.items[idx]


class SyntheticDataModule(VitalDataModule):
    """Data module generating random patients whose attributes all depend on a shared latent vector.

    Every attribute declared in the schema is generated from a low-dimensional latent vector specific to each patient,
    so that the attributes are correlated and the targets can be predicted from the other attributes.
    """

    def __init__(
        self,
        patients_kwargs: Optional[Mapping[str, Any]] = None,
        process_patient_kwargs: Optional[Mapping[str, Any]] = None,
        num_patients: Optional[Mapping[str, int]] = None,
        time_series_len: int = 64,
        latent_dim: int = 4,
        missing_rate: float = 0.1,
        noise_std: float = 0.3,
        seed: int = 0,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            patients_kwargs: Parameters describing the patients. Only `views` is used, to select the views for which to
                generate time-series attributes (defaults to all the views in the schema).
            process_patient_kwargs: Parameters describing the items. Only `tabular_attrs` and `time_series_attrs` are
                used, to select the attributes to include in the items (default to all the attributes in the schema).
            num_patients: Number of patients to generate in each subset.
            time_series_len: Number of samples in each time-series attribute.
            latent_dim: Dimensionality of the latent vector from which the attributes of each patient are generated.
            missing_rate: Probability for each tabular attribute of a patient to be missing.
            noise_std: Standard deviation of the noise added to each attribute.
            seed: Seed of the random generator, so that the same dataset is generated every time.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        patients_kwargs = patients_kwargs or {}
        process_patient_kwargs = process_patient_kwargs or {}
        self.views = [View[view] for view in patients_kwargs.get("views") or View]
        self.tabular_attrs = [
            TabularAttribute[attr] for attr in process_patient_kwargs.get("tabular_attrs") or TabularAttribute
        ]
        time_series_attrs = process_patient_kwargs.get("time_series_attrs")
        if time_series_attrs is None:
            time_series_attrs = TimeSeriesAttribute
        self.time_series_attrs = [TimeSeriesAttribute[attr] for attr in time_series_attrs]

        self.num_patients = {Subset.TRAIN: 512, Subset.VAL: 128, Subset.TEST: 128}
        self.num_patients.update({Subset(subset): num for subset, num in (num_patients or {}).items()})
        self.time_series_len = time_series_len
        self.latent_dim = latent_dim
        self.missing_rate = missing_rate
        self.noise_std = noise_std

        # Draw the (fixed) mappings from the latent space to each attribute
        rng = np.random.default_rng(seed)
        self._tab_weights = {attr: rng.normal(size=latent_dim) / np.sqrt(latent_dim) for attr in TabularAttribute}
        self._ts_weights = {
            (view, attr): rng.normal(size=(3, latent_dim)) / np.sqrt(latent_dim)
            for view in View
            for attr in TimeSeriesAttribute
        }
        self._subsets_seeds = {subset: seed + idx + 1 for idx, subset in enumerate(Subset)}
        self.subsets_patients: Dict[Subset, Dict[str, PatientRecord]] = {}  # To populate in `setup`

        in_shape = {}
        if self.tabular_attrs:
            in_shape[DataTag.tabular_attrs] = (len(self.tabular_attrs), 1)
        if self.views and self.time_series_attrs:
            in_shape[DataTag.time_series_attrs] = (len(self.views) * len(self.time_series_attrs), time_series_len)
        super().__init__(data_params=DataParameters(in_shape=in_shape), **kwargs)

    def _generate_patient(self, patient_id: str, rng: np.random.Generator) -> PatientData:
        z = rng.normal(size=self.latent_dim)
        item: PatientData = {"id": patient_id}

        for attr in self.tabular_attrs:
            score = self._tab_weights[attr] @ z + rng.normal(scale=self.noise_std)
            is_missing = rng.random() < self.missing_rate
            if attr in TABULAR_CAT_ATTR_LABELS:
                # Split the (standard normal) scores into classes of equal probability
                num_classes = len(TABULAR_CAT_ATTR_LABELS[attr])
                bins = stats.norm.ppf(np.linspace(0, 1, num_classes + 1)[1:-1], scale=np.sqrt(1 + self.noise_std**2))
                label = MISSING_CAT_ATTR if is_missing else int(np.digitize(score, bins))
                item[attr] = np.array(label, dtype=np.int64)
            else:
                item[attr] = np.array(MISSING_NUM_ATTR if is_missing else score, dtype=np.float32)

        t = np.linspace(0, 1, self.time_series_len)
        for view in self.views:
            item[view] = {}
            for attr in self.time_series_attrs:
                amplitude, phase, offset = self._ts_weights[(view, attr)] @ z
                values = (1 + 0.5 * np.tanh(amplitude)) * np.sin(2 * np.pi * t + phase) + offset
                values += rng.normal(scale=self.noise_std / 4, size=self.time_series_len)
                item[view][attr] = values.astype(np.float32)

        return item

    def _build_subset_dataset(self, subset: Subset) -> Dataset:
        rng = np.random.default_rng(self._subsets_seeds[subset])
        items = [self._generate_patient(f"{subset}_{idx:04d}", rng) for idx in range(self.num_patients[subset])]
        self.subsets_patients[subset] = {item["id"]: self._to_record(item) for item in items}
        return _ItemsDataset(items)

    def _to_record(self, item: PatientData) -> PatientRecord:
        attrs = {}
        for attr in self.tabular_attrs:
            if attr in TABULAR_CAT_ATTR_LABELS:
                if (label_idx := int(item[attr])) != MISSING_CAT_ATTR:
                    attrs[attr] = TABULAR_CAT_ATTR_LABELS[attr][label_idx]
            elif not np.isnan(item[attr]):
                attrs[attr] = float(item[attr])
        return PatientRecord(id=item["id"], attrs=attrs)

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        subsets_to_setup = []
        match stage:
            case TrainerFn.FITTING:
                subsets_to_setup.extend([Subset.TRAIN, Subset.VAL])
            case TrainerFn.VALIDATING:
                subsets_to_setup.append(Subset.VAL)
            case TrainerFn.TESTING:
                subsets_to_setup.append(Subset.TEST)
            case TrainerFn.PREDICTING:
                subsets_to_setup.extend(PREDICT_DATALOADERS_SUBSETS)
        self.datasets.update({subset: self._build_subset_dataset(subset) for subset in subsets_to_setup})

    def predict_dataloader(self) -> Sequence[DataLoader]:  # noqa: D102
        return [
            DataLoader(self.datasets[subset], batch_size=None, num_workers=self.num_workers)
            for subset in PREDICT_DATALOADERS_SUBSETS
        ]

