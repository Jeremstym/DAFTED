import csv
import itertools
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Hashable, List, Callable, Iterable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dataprocessing.data.orchid.config import TabularAttribute, TimeSeriesAttribute
from dataprocessing.data.orchid.config import View as ViewEnum
from dataprocessing.data.orchid.data_module import PREDICT_DATALOADERS_SUBSETS
from dataprocessing.data.orchid.datapipes import PatientData, filter_time_series_attributes
from dataprocessing.data.orchid.utils.attributes import (
    TABULAR_CAT_ATTR_LABELS,
    build_attributes_dataframe,
    plot_attributes_wrt_time,
)
from dataprocessing.data.config import Subset
from dataprocessing.utils.loggers import log_dataframe, log_figure
from dataprocessing.utils.plot import embedding_scatterplot
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.callbacks.prediction_writer import WriteInterval
from pytorch_lightning import Trainer
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize  # type: ignore[import-untyped]
from torch import Tensor
from tqdm import tqdm

from didactic.tasks.cardiac_multimodal_tabpfn import CardiacMultimodalTabPFN


class CardiacRepresentationPredictionWriter(BasePredictionWriter):
    """Prediction writer that measures prediction performance for cardiac representation tasks."""

    def __init__(
        self,
        write_path: Optional[Union[str, Path]] = None,
        hue_attrs: Optional[Sequence[str]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initializes class instance.

        Args:
            write_path: Root directory under which to save the predictions / analysis plots.
            hue_attrs: Attributes to display the scatter plot w.r.t.
            embedding_kwargs: Parameters to pass along to the PaCMAP embedding.
        """
        super().__init__(write_interval=WriteInterval.EPOCH)  # type: ignore[arg-type]
        self._write_path = Path(write_path) if write_path else None
        self._hue_attrs = hue_attrs if hue_attrs else []
        self._embedding_kwargs = {} if embedding_kwargs is None else embedding_kwargs

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Removes results potentially left behind by previous runs of the callback in the same directory."""
        # Write to the same directory as the experiment logger if no custom path is provided
        if self._write_path is None:
            self._write_path = pl_module.log_dir / "predictions"

        # Delete leftover predictions from previous run
        shutil.rmtree(self._write_path, ignore_errors=True)

        # Ensure that matplotlib is using 'agg' backend
        # to avoid possible leak of file handles if matplotlib defaults to another backend
        plt.switch_backend("agg")

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: CardiacMultimodalTabPFN,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        """Measures and saves prediction performance for cardiac representation tasks.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            predictions: Sequences of encoder output features and predicted tabular attributes for each patient. There
                is one sublist for each prediction dataloader provided.
            batch_indices: Indices of all the batches whose outputs are provided.
        """
        assert self._write_path is not None, "Write path must be set before writing predictions."
        self._write_path.mkdir(parents=True, exist_ok=True)

        # If the model includes prediction heads to predict attributes from the features, analyze the prediction results
        if pl_module.prediction_heads:
            self._write_prediction_scores(trainer, pl_module, predictions)

    def _write_prediction_scores(
        self,
        trainer: "pl.Trainer",
        pl_module: CardiacMultimodalTabPFN,
        predictions: Sequence[Any],
        # batch_indices: Optional[Sequence[Any]],
    ) -> None:
        """Measure and save prediction scores.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            predictions: Sequences of encoder output features and predicted tabular attributes for each patient. There
                is one sublist for each prediction dataloader provided.
        """
        target_categorical_attrs = [
            attr for attr in pl_module.hparams["predict_losses"] if attr in TabularAttribute.categorical_attrs()
        ]
        target_numerical_attrs = [
            attr for attr in pl_module.hparams["predict_losses"] if attr in TabularAttribute.numerical_attrs()
        ]

        for subset, subset_predictions in zip([Subset.TEST, Subset.TRAIN], predictions):
            # Compute metrics on the predictions for all the patients of the subset +
            # collect and structure necessary predictions to compute these metrics
            if subset != Subset.TEST:
                continue
            subset_patients = trainer.datamodule.subsets_patients[subset]
            subset_categorical_data, subset_numerical_data = [], []
            classification_out = {attr: [] for attr in target_categorical_attrs}
            for (patient_id, patient), patient_predictions in zip(subset_patients.items(), subset_predictions):
                attr_predictions = patient_predictions[1]
                if target_categorical_attrs:
                    patient_categorical_data = {"patient": patient_id}
                    for attr in target_categorical_attrs:
                        # Collect the classification logits/probabilities
                        classification_out.setdefault(attr, []).append(attr_predictions[attr].detach().cpu().numpy())
                        # Add the hard prediction and target labels
                        patient_categorical_data.update(
                            {
                                f"{attr}_prediction": TABULAR_CAT_ATTR_LABELS[attr][attr_predictions[attr].argmax()],
                                f"{attr}_target": patient.attrs.get(attr, np.nan),
                                f"{attr}_probas": attr_predictions[attr].detach().cpu().numpy(),
                            }
                        )
                    subset_categorical_data.append(patient_categorical_data)

            # Convert the classification logits/probabilities to numpy arrays
            for attr, attr_pred in classification_out.items():
                # Convert to numpy array and ensure float32, to avoid numerical instabilities in case of float16 values
                # coming from AMP models. This is especially important for softmax, which is sensitive to small values.
                prediction = np.array(attr_pred, dtype=np.float32)
                # if (attr_pred < 0).any() or (attr_pred > 1).any():
                if not (prediction.sum(axis=1) == 1).all():
                    # If output were logits, compute probabilities from logits
                    probits = softmax(prediction, axis=1)
                classification_out[attr] = probits  # type: ignore[assignment]

            if subset_categorical_data:
                subset_categorical_df = pd.DataFrame.from_records(subset_categorical_data, index="patient")
                subset_categorical_stats = subset_categorical_df.describe().drop(["count"])

                # Compute additional custom metrics (i.e. not reported by `describe`) for categorical attributes
                notna_mask = subset_categorical_df.notna()
                for attr in target_categorical_attrs:
                    # Extract target labels as well as predicted labels and probabilities from saved outputs
                    target = subset_categorical_df[f"{attr}_target"][notna_mask[f"{attr}_target"]]
                    pred_labels = subset_categorical_df[f"{attr}_prediction"][notna_mask[f"{attr}_target"]]
                    pred_probas = classification_out[attr][notna_mask[f"{attr}_target"]]

                    # Compute ordered numerical labels from saved outputs
                    labels_arr = np.array(TABULAR_CAT_ATTR_LABELS[attr], ndmin=2)
                    target_num_labels = (target.to_numpy().reshape(-1, 1) == labels_arr).argmax(axis=1)

                    labels_list = TABULAR_CAT_ATTR_LABELS[attr]

                    # Compute metrics
                    subset_categorical_stats.loc["acc", f"{attr}_prediction"] = accuracy_score(target, pred_labels)
                    subset_categorical_stats.loc["auroc", f"{attr}_prediction"] = roc_auc_score(
                        target_num_labels, pred_probas, multi_class="ovr"
                    )
                    subset_categorical_stats.loc["f1_avg", f"{attr}_prediction"] = f1_score(
                        target, pred_labels, average="macro"
                    )
                    subset_categorical_stats.loc["auprc_avg", f"{attr}_prediction"] = average_precision_score(
                        target_num_labels, pred_probas, average="macro"
                    )
                    subset_categorical_stats.loc["confusion_matrix", f"{attr}_prediction"] = confusion_matrix(
                        target,
                        pred_labels,
                        labels=labels_list
                    ).tolist()

                    y_bins = label_binarize(target_num_labels, classes=np.arange(len(labels_arr[0])))
                    auroc_scores = {}
                    for i, label in enumerate(labels_arr[0]):
                        auroc_scores[label] = roc_auc_score(y_bins[:, i], pred_probas[:, i])
                        subset_categorical_stats.loc[f"auroc_{label}", f"{attr}_prediction"] = auroc_scores[label]

                # Concatenate the element-wise results + statistics in one dataframe
                subset_categorical_scores = pd.concat([subset_categorical_stats, subset_categorical_df])

                if subset_numerical_data:
                    subset_numerical_df = pd.DataFrame.from_records(subset_numerical_data, index="patient")
                    subset_numerical_stats = subset_numerical_df.describe(percentiles=[]).drop(["count"])
                    # Compute additional custom metrics (i.e. not reported by `describe`) for numerical attributes
                    notna_mask = subset_numerical_df.notna()
                    subset_numerical_stats.loc["mae"] = {
                        f"{attr}_prediction": mean_absolute_error(
                            subset_numerical_df[f"{attr}_target"][notna_mask[f"{attr}_target"]],
                            subset_numerical_df[f"{attr}_prediction"][notna_mask[f"{attr}_target"]],
                        )
                        for attr in target_numerical_attrs
                    }
                    subset_numerical_stats.loc["corr"] = {
                        f"{attr}_prediction": float(
                            stats.pearsonr(
                                subset_numerical_df[f"{attr}_target"][notna_mask[f"{attr}_target"]],
                                subset_numerical_df[f"{attr}_prediction"][notna_mask[f"{attr}_target"]],
                            ).statistic
                        )
                        for attr in target_numerical_attrs
                    }

                    # Concatenate the element-wise results + statistics in one dataframe
                    subset_numerical_scores = pd.concat([subset_numerical_stats, subset_numerical_df])

                # Log the prediction scores and statistics using the experiment logger
                prediction_scores_to_log = {}
                if subset_categorical_data:
                    prediction_scores_to_log["categorical"] = subset_categorical_scores
                if subset_numerical_data:
                    prediction_scores_to_log["numerical"] = subset_numerical_scores
                if pl_module.hparams["explainability"]:
                    prediction_scores_to_log["explainable"] = df_latent_norm
                for tag, prediction_scores in prediction_scores_to_log.items():
                    # Log the prediction scores to the (online) experiment logger
                    assert self._write_path is not None, "Write path must be set before writing predictions."
                    data_filepath = self._write_path / f"{subset}_{tag}_scores.csv"
                    # log_dataframe(trainer.logger, prediction_scores, filename=data_filepath.name)

                    # Save the prediction scores locally
                    prediction_scores.to_csv(data_filepath, quoting=csv.QUOTE_NONNUMERIC)
