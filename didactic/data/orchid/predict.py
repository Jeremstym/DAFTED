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
from dataprocessing.utils.loggers import log_dataframe, log_figure
from dataprocessing.utils.plot import embedding_scatterplot
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.callbacks.prediction_writer import WriteInterval
from pytorch_lightning import Trainer
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import ( #type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize # type: ignore[import-untyped]
from torch import Tensor
from tqdm import tqdm

from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask
from didactic.tasks.cardiac_sequence_attrs_ae import CardiacSequenceAttributesAutoencoder


class CardiacSequenceAttributesPredictionWriter(BasePredictionWriter):
    """Prediction writer that plots reconstructed time-series attributes and plots the latent space manifold."""

    def __init__(self, write_path: Optional[Union[str, Path]] = None, embedding_kwargs: Optional[Dict[str, Any]] = None):
        """Initializes class instance.

        Args:
            write_path: Root directory under which to save the predictions / analysis plots.
            embedding_kwargs: Parameters to pass along to the PaCMAP embedding.
        """
        super().__init__(write_interval=WriteInterval.BATCH_AND_EPOCH) #type: ignore[arg-type]
        self._write_path = Path(write_path) if write_path else None
        self._embedding_kwargs = {} if embedding_kwargs is None else embedding_kwargs

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Removes results potentially left behind by previous runs of the callback in the same directory."""
        # Write to the same directory as the experiment logger if no custom path is provided
        if self._write_path is None:
            self._write_path = pl_module.log_dir / "predictions_plots"

        # Assign a subdirectory for each dataloader/subset to predict on
        self._dataloaders_write_path = [self._write_path / subset for subset in PREDICT_DATALOADERS_SUBSETS]

        # Delete leftover predictions from previous run
        shutil.rmtree(self._write_path, ignore_errors=True)

        # Ensure that matplotlib is using 'agg' backend
        # to avoid possible leak of file handles if matplotlib defaults to another backend
        plt.switch_backend("agg")

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: CardiacSequenceAttributesAutoencoder,
        prediction: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tuple[Tensor, Tensor]],
        batch_indices: Optional[Sequence[int]],
        batch: PatientData,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Saves plots of the attributes' reconstructed curves vs their input curves.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            prediction: Mapping between attributes' keys and tuples of i) their reconstructions and ii) their encodings
                in the latent space.
            batch_indices: Indices of all the batches whose outputs are provided.
            batch: The current batch used by the model to give its prediction.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
        """
        patient_id = list(trainer.datamodule.subsets_patients[PREDICT_DATALOADERS_SUBSETS[dataloader_idx]])[batch_idx] # type: ignore[attr-defined]

        # Collect the attributes predictions and convert them to numpy arrays
        time_series_attrs_reconstructions = {
            attr_key: attr_prediction[0].cpu().numpy() for attr_key, attr_prediction in prediction.items()
        }
        # Collect the attributes data and convert it to numpy arrays,
        # only keeping the attributes for which we have predictions
        time_series_attrs = {
            attr_key: attr.cpu().numpy() if isinstance(attr, torch.Tensor) else attr
            for attr_key, attr in filter_time_series_attributes(batch)[0].items()
            if attr_key in time_series_attrs_reconstructions
        }
        attrs = {"data": time_series_attrs, "pred": time_series_attrs_reconstructions}

        # Plot the curves for each attribute w.r.t. time
        attrs_df = build_attributes_dataframe(attrs, normalize_time=True) # type: ignore[arg-type]
        for title, plot in plot_attributes_wrt_time(attrs_df, plot_title_root=patient_id):
            batch_dir = self._dataloaders_write_path[dataloader_idx] / patient_id
            batch_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(batch_dir / f"{title}.png")
            plt.close()  # Close the figure to avoid contamination between plots

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: CardiacSequenceAttributesAutoencoder,
        predictions: Sequence[Sequence[Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tuple[Tensor, Tensor]]]],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        """Saves plots of the distribution of attributes' encodings.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            predictions: Sequences of predictions for each patient, with the content of the predictions for each patient
                detailed in the docstring for `write_on_batch_end`. There is one sublist for each prediction dataloader
                provided.
            batch_indices: Indices of all the batches whose outputs are provided.
        """
        # Build a dataframe for the encodings of the whole dataset, with some metadata about each encoding to be able
        # to visualize the distribution of encodings w.r.t. this metadata
        encodings = {
            (subset, patient_id, *attr_key): attr_predictions[1].cpu().numpy()
            # For each prediction dataloader
            for subset, subset_predictions in zip(PREDICT_DATALOADERS_SUBSETS, predictions)
            # For each batch of data in a dataloader
            for patient_id, patient_predictions in zip(trainer.datamodule.subsets_patients[subset], subset_predictions) # type: ignore[attr-defined]
            # For each attribute prediction in the batch
            for attr_key, attr_predictions in patient_predictions.items()
        }
        encodings_df = pd.DataFrame(
            encodings.values(),
            index=pd.MultiIndex.from_tuples(encodings.keys(), names=["subset", "patient", "view", "attr"]),
        )

        plots = {
            "latent_space_by_attrs": {"hue": "attr", "style": "view"},
            "latent_space_by_subsets": {"hue": "subset"},
        }
        for plot_filename, _ in zip(
            plots,
            embedding_scatterplot(encodings_df, plots.values(), data_tag="latent space", **self._embedding_kwargs),
        ):
            # Log the plots using the experiment logger
            # log_figure(trainer.logger, figure_name=plot_filename)

            # Save the plots locally
            assert self._write_path is not None, "Write path must be set before writing predictions."
            plt.savefig(self._write_path / f"{plot_filename}.png")
            plt.close()  # Close the figure to avoid contamination between plots


class CardiacRepresentationPredictionWriter(BasePredictionWriter):
    """Prediction writer that measures prediction performance for cardiac representation tasks."""

    def __init__(
        self, write_path: Optional[Union[str, Path]] = None, hue_attrs: Optional[Sequence[str]] = None, embedding_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Initializes class instance.

        Args:
            write_path: Root directory under which to save the predictions / analysis plots.
            hue_attrs: Attributes to display the scatter plot w.r.t.
            embedding_kwargs: Parameters to pass along to the PaCMAP embedding.
        """
        super().__init__(write_interval=WriteInterval.EPOCH) #type: ignore[arg-type]
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
        pl_module: CardiacMultimodalRepresentationTask,
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
        # self._write_features_plots(trainer, pl_module, predictions)

        # If the model includes prediction heads to predict attributes from the features, analyze the prediction results
        if pl_module.prediction_heads:
            self._write_prediction_scores(trainer, pl_module, predictions, batch_indices)

    def _write_features_plots(
        self, trainer: "pl.Trainer", pl_module: CardiacMultimodalRepresentationTask, predictions: Sequence[Any]
    ) -> None:
        """Plots the distribution of features learned by the encoder w.r.t. tabular attributes and some metadata.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            predictions: Sequences of encoder output features and predicted tabular attributes for each patient. There
                is one sublist for each prediction dataloader provided.
        """
        prediction_example = predictions[0][0]  # 1st: subset, 2nd: batch
        # Pre-compute the list of attributes for which we have a continuum parameter, since this output might be None
        # and we don't want to access it in that case
        ordinal_attrs = list(prediction_example[2]) if prediction_example[2] else []
        features = {
            (
                subset,
                patient.id,
                *[patient.attrs.get(attr) for attr in self._hue_attrs],
                *[patient_prediction[output_idx][attr].item() for attr in ordinal_attrs for output_idx in (2, 3)],
            ): patient_prediction[0]
            .flatten()
            .cpu()
            .numpy()
            # For each prediction dataloader
            for subset, subset_predictions in zip(PREDICT_DATALOADERS_SUBSETS, predictions)
            # For each batch of data in a dataloader
            for patient, patient_prediction in zip(
                trainer.datamodule.subsets_patients[subset].values(), subset_predictions # type: ignore[attr-defined]
            )
        }
        features_df = pd.DataFrame(
            features.values(),
            index=pd.MultiIndex.from_tuples(
                features.keys(),
                names=[
                    "subset",
                    "patient",
                    *self._hue_attrs,
                    *[f"{attr}_continuum_{pred_desc}" for attr in ordinal_attrs for pred_desc in ("param", "tau")],
                ],
            ),
        )

        # Plot data w.r.t. all indexing data, except for specific patient
        plots = {
            f"features_wrt_{index_name}": {
                "hue": index_name,
                "hue_order": TABULAR_CAT_ATTR_LABELS.get(index_name), # type: ignore[call-overload]
            }
            for index_name in features_df.index.names
            if index_name != "patient"
        }
        for plot_filename, _ in zip(
            plots,
            embedding_scatterplot(features_df, plots.values(), data_tag="features", **self._embedding_kwargs),
        ):
            # Log the plots using the experiment logger
            # log_figure(trainer.logger, figure_name=plot_filename)

            # Save the plots locally
            assert self._write_path is not None, "Write path must be set before writing predictions."
            plt.savefig(self._write_path / f"{plot_filename}.png")
            plt.close()  # Close the figure to avoid contamination between plots

        if pl_module.hparams["latent_representation"]:
            assert pl_module.hparams["rpr_method"] in [
                "tsne",
                "umap",
                "pacmap",
            ], f"Unknown embedding method '{pl_module.hparams['rpr_method']}'. Must be one of: ['tsne', 'umap', 'pacmap']."
            method = pl_module.hparams['rpr_method']
            feature_latent = {
                (subset, patient.id, patient.attrs.get(attr), data_type): patient_prediction[4][
                    data_type
                ]  # Accessing the prediction based on data type
                .flatten()
                .cpu()
                .numpy()
                for subset, subset_predictions in zip(PREDICT_DATALOADERS_SUBSETS, predictions)
                for patient, patient_prediction in zip(
                    trainer.datamodule.subsets_patients[subset].values(), subset_predictions # type: ignore[attr-defined]
                )
                for data_type in list(patient_prediction[4].keys())  # Iterating over data types
                # for data_type in ["tabular common","tabular unique","time-series"]  # Iterating over data types
                for attr in pl_module.hparams["predict_losses"]  # Iterating over target attributes
            }
            # Create a MultiIndex from the keys of the feature_latent dictionary
            multi_index = pd.MultiIndex.from_tuples(
                feature_latent.keys(), names=["Subset", "Patient ID", "Label", "Modality Type"]
            )

            # Create the DataFrame using the values and the MultiIndex
            df_latent = pd.DataFrame(
                list(feature_latent.values()), index=multi_index  # Convert values to a list  # Set the MultiIndex
            )

            # Plot data w.r.t. all indexing data, except for specific patient
            plots_latent = {
                f"latent_space_wrt_{index_name}": {
                    "hue": index_name,
                    "hue_order": TABULAR_CAT_ATTR_LABELS.get( # type: ignore[call-overload]
                        index_name
                    ),  # Use categorical attrs' predefined labels order
                }
                for index_name in df_latent.index.names
                if index_name != "Patient ID"
            }
            if pl_module.hparams["rpr_method"] == "tsne":
                self._embedding_kwargs["perplexity"] = 30
                # self._embedding_kwargs["learning_rate"] = 200
                self._embedding_kwargs["random_state"] = 42
            elif pl_module.hparams["rpr_method"] == "pacmap":
                self._embedding_kwargs["n_neighbors"] = 5
                self._embedding_kwargs["MN_ratio"] = 1.0
                self._embedding_kwargs["num_iters"] = 25

            for plot_filename, _ in zip(
                plots_latent,
                embedding_scatterplot(
                    df_latent, plots_latent.values(), data_tag="latent space", method=method, **self._embedding_kwargs
                ),
            ):
                # Log the plots using the experiment logger
                # log_figure(trainer.logger, figure_name=plot_filename)
                # Increase legend font size
                # plt.legend(fontsize='large', title_fontsize='xx-large')
                # Save the plots locally
                assert self._write_path is not None, "Write path must be set before writing predictions."
                plt.savefig(self._write_path / f"{plot_filename}.png")
                plt.close()

    def _write_prediction_scores(
        self,
        trainer: "pl.Trainer",
        pl_module: CardiacMultimodalRepresentationTask,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
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
        for subset, subset_predictions in zip(PREDICT_DATALOADERS_SUBSETS[PREDICT_DATALOADERS_SUBSETS.index("test"):], predictions):
            subset_patients = trainer.datamodule.subsets_patients[subset] # type: ignore[attr-defined]

            # Compute metrics on the predictions for all the patients of the subset +
            # collect and structure necessary predictions to compute these metrics
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

            # subset_categorical_data: List[Dict[str, Any]] = []
            # subset_numerical_data: List[Dict[str, Any]] = []
            # classification_out: Dict[str, List[np.ndarray]] = {attr: [] for attr in target_categorical_attrs}
            # for (patient_id, patient), patient_predictions in itertools.product(
            #     subset_patients.items(), subset_predictions
            # ):
            #     attr_predictions = patient_predictions[1]
            #     if target_categorical_attrs:
            #         patient_categorical_data = {"patient": patient_id}
            #         for attr in target_categorical_attrs:
            #             if True:
            #                 # Collect the classification logits/probabilities
            #                 print(attr_predictions)
            #                 print(type(attr_predictions))
                            # attr_prediction = attr_predictions[patient_id]  # Get the prediction for the current patient
                            # classification_out.setdefault(attr, []).append(attr_prediction.detach().cpu().numpy())
                            # # Add the hard prediction and target labels
                            # patient_categorical_data.update(
                            #     {
                            #         f"{attr}_prediction": TABULAR_CAT_ATTR_LABELS[attr][attr_prediction.argmax()],
                            #         f"{attr}_target": patient.attrs.get(attr, np.nan),
                            #         f"{attr}_probas": attr_prediction.detach().cpu().numpy(),
                            #     }
                            # )
                            # subset_categorical_data.append(patient_categorical_data)
                        # else:
                        #     # Collect the classification logits/probabilities
                        #     classification_out.setdefault(attr, []).append(
                        #         attr_predictions[attr].detach().cpu().numpy()
                        #     )
                        #     # Add the hard prediction and target labels
                        #     patient_categorical_data.update(
                        #         {
                        #             f"{attr}_prediction": TABULAR_CAT_ATTR_LABELS[attr][
                        #                 attr_predictions[attr].argmax()
                        #             ],
                        #             f"{attr}_target": patient.attrs.get(attr, np.nan),
                        #             f"{attr}_probas": attr_predictions[attr].detach().cpu().numpy(),
                        #         }
                        #     )
                        #     subset_categorical_data.append(patient_categorical_data)

                # if target_numerical_attrs:
                #     patient_numerical_data = {"patient": patient_id}
                #     for attr in target_numerical_attrs:
                #         patient_numerical_data.update(
                #             {
                #                 f"{attr}_prediction": attr_predictions[attr].item(),
                #                 f"{attr}_target": patient.attrs.get(attr, np.nan),
                #             }
                #         )
                #     subset_numerical_data.append(patient_numerical_data)

            # Convert the classification logits/probabilities to numpy arrays
            for attr, attr_pred in classification_out.items():
                # Convert to numpy array and ensure float32, to avoid numerical instabilities in case of float16 values
                # coming from AMP models. This is especially important for softmax, which is sensitive to small values.
                prediction = np.array(attr_pred, dtype=np.float32)
                # if (attr_pred < 0).any() or (attr_pred > 1).any():
                if not (prediction.sum(axis=1) == 1).all():
                    # If output were logits, compute probabilities from logits
                    probits = softmax(prediction, axis=1)
                classification_out[attr] = probits # type: ignore[assignment]

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

                    # labels_list = [0, 1, 3, 2, 4]

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
                        target, pred_labels,
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
                    f"{attr}_prediction": float(stats.pearsonr(
                        subset_numerical_df[f"{attr}_target"][notna_mask[f"{attr}_target"]],
                        subset_numerical_df[f"{attr}_prediction"][notna_mask[f"{attr}_target"]],
                    ).statistic)
                    for attr in target_numerical_attrs
                }

                # Concatenate the element-wise results + statistics in one dataframe
                subset_numerical_scores = pd.concat([subset_numerical_stats, subset_numerical_df])

            if pl_module.hparams["explainability"]:
                feature_explainable = {
                    (subset, patient.id, patient.attrs.get(attr), latent_token): torch.norm(
                        patient_prediction[5][latent_token]
                    )  # Accessing the prediction based on data type
                    .flatten()
                    .cpu()
                    .numpy()
                    for subset, subset_predictions in zip(PREDICT_DATALOADERS_SUBSETS, predictions)
                    for patient, patient_prediction in zip(
                        trainer.datamodule.subsets_patients[subset].values(), subset_predictions # type: ignore[attr-defined]
                    )
                    for latent_token in list(patient_prediction[5].keys())  # Iterating over data types
                    for attr in pl_module.hparams["predict_losses"]  # Iterating over target attributes
                }
                # Create a MultiIndex from the keys of the feature_latent dictionary
                multi_index = pd.MultiIndex.from_tuples(
                    feature_explainable.keys(), names=["Subset", "Patient ID", "Label", "Token"]
                )

                # Create the DataFrame using the values and the MultiIndex
                df_latent_norm = (
                    pd.DataFrame(
                        list(feature_explainable.values()),  # Convert values to a list
                        index=multi_index,  # Set the MultiIndex
                    )
                    .groupby("Token")
                    .mean()
                )

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
