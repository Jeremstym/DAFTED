import itertools
import logging
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from seaborn import PairGrid
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from dataprocessing.data.cardinal.config import CardinalTag, TabularAttribute, TimeSeriesAttribute
from dataprocessing.data.cardinal.config import View as ViewEnum
from dataprocessing.data.cardinal.utils.attributes import (
    TABULAR_ATTR_UNITS,
    TABULAR_CAT_ATTR_LABELS,
    TIME_SERIES_ATTR_LABELS,
    build_attributes_dataframe,
)
from dataprocessing.data.cardinal.utils.data_struct import Patient
from dataprocessing.data.cardinal.utils.itertools import Patients
from dataprocessing.data.transforms import Interp1d

logger = logging.getLogger(__name__)


def check_subsets(patients: Sequence[Patient.Id], subsets: Dict[str, Sequence[Patient.Id]]) -> None:
    """Checks the lists of patients overall and in each subset to ensure each patient belongs to one and only subset.

    Args:
        patients: Collection of patients.
        subsets: Lists of patients making up subsets.
    """
    for (subset1, subset1_patients), (subset2, subset2_patients) in itertools.combinations(subsets.items(), 2):
        if intersect := set(subset1_patients) & set(subset2_patients):
            raise RuntimeError(
                f"All provided subsets should be disjoint from each other, but subsets {subset1}' and '{subset2}' "
                f"have the following patients in common: {sorted(intersect)}."
            )

    patient_ids_in_subsets = set().union(*subsets.values())
    if unassigned_patients := set(patients) - patient_ids_in_subsets:
        raise RuntimeError(
            f"All patients should be part of one of the subset. However, the following patients are not included in "
            f"any subset: {sorted(unassigned_patients)}."
        )


def generate_patients_splits(
    patients: Patients,
    stratify: TabularAttribute,
    bins: int = 5,
    test_size: Union[int, float] = None,
    train_size: Union[int, float] = None,
    seed: int = None,
    progress_bar: bool = False,
) -> Tuple[List[Patient.Id], List[Patient.Id]]:
    """Splits patients into train and test subsets, preserving the distribution of `stratify` variable across subsets.

    Notes:
        - Wrapper around `sklearn.model_selection.train_test_split` that performs binning on continuous variables, since
          out-of-the-box `sklearn`'s `train_test_split` only works with categorical `stratify` variables.

    Args:
        patients: Collection of patients to split.
        stratify: Name of the tabular attribute whose distribution in each of the subset should be similar. Contrary to
            `sklearn.model_selection.train_test_split`, this attribute can be continuous.
        bins: If `stratify` is a continuous attribute, number of bins into which to categorize the values, to ensure
            each bin is distributed representatively in the split.
        test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the
            test split. If int, represents the absolute number of test samples. If None, the value is set to the
            complement of the train size.
        train_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in
            the train split. If int, represents the absolute number of train samples. If None, the value is
            automatically set to the complement of the test size.
        seed: Seed to control the shuffling applied to the data before applying the split.
        progress_bar: If ``True``, enables progress bars detailing the progress of the collecting data from patients.

    Returns:
        Lists of patients in the train and tests subsets, respectively.
    """
    patients = patients.values()
    msg = "Collecting patients' data"
    if progress_bar:
        patients = tqdm(patients, desc=msg, unit="patient")
    else:
        logger.info(msg + "...")

    # Collect the data of the attribute by which to stratify the split from the patient
    patients_stratify = {patient.id: patient.attrs[stratify] for patient in patients}

    if stratify in TabularAttribute.numerical_attrs():
        # Compute categorical stratify variable from scalar attribute
        stratify_vals = list(patients_stratify.values())
        stratify_bins = np.linspace(min(stratify_vals), max(stratify_vals), num=bins + 1)
        stratify_bins[-1] += 1e-6  # Add epsilon to the last bin's upper bound since it's excluded by `np.digitize`
        stratify_labels = np.digitize(stratify_vals, stratify_bins) - 1  # Subtract 1 because bin indexing starts at 1
    else:
        stratify_labels = list(patients_stratify.values())

    logger.info("Generating splits...")
    patient_ids_train, patient_ids_test = train_test_split(
        list(patients_stratify), test_size=test_size, train_size=train_size, random_state=seed, stratify=stratify_labels
    )
    return sorted(patient_ids_train), sorted(patient_ids_test)
