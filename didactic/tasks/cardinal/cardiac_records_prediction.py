import itertools
import logging
from typing import Dict, Sequence, Tuple, Optional, List

import hydra
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from dataprocessing.data.config import Subset
from dataprocessing.data.cardinal.config import CardinalTag, TabularAttribute, TimeSeriesAttribute
from dataprocessing.data.cardinal.config import View as ViewEnum
from dataprocessing.data.cardinal.data_module import OrchidDataModule
from dataprocessing.data.cardinal.datapipes import MISSING_CAT_ATTR, PatientData, filter_time_series_attributes
from dataprocessing.utils.config import register_omegaconf_resolvers
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.base import ClassifierMixin # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class CardiacRecordsPredictionTask:
    """XGBoost|TabPFN model for diagnosis prediction from EHR tabular data."""

    def __init__(
        self,
        model: ClassifierMixin,
        tabular_attrs: Sequence[TabularAttribute],
        target_attr: TabularAttribute,
        time_series_attrs: Optional[Sequence[TimeSeriesAttribute]] = None,
        views: Sequence[ViewEnum] = tuple(ViewEnum),
        use_time_series: bool = False,
        save_model: bool = True,
    ):
        """Initializes class instance.

        Args:
            model: Generic classifier model implementing the `sklearn` API.
            tabular_attrs: List of tabular attributes to use as input features for the classifier.
            target_attr: Tabular attribute to use as the target label for the classifier.
        """
        for method in ["fit", "predict_proba"]:
            if not callable(getattr(model, method, None)):
                raise ValueError(f"Model must implement method: {method}")

        self.model = model

        # Ensure string tags are converted to their appropriate enum types
        self.tabular_attrs = tuple(TabularAttribute[e] for e in tabular_attrs)
        self.target_attr = TabularAttribute[target_attr]
        self.save_model = save_model
        if use_time_series:
            assert time_series_attrs is not None, "Time series attributes must be provided if use_time_series is True."
            self.use_time_series = True
            self.views = tuple(ViewEnum[e] for e in views)
            self.time_series_attrs = tuple(TimeSeriesAttribute[e] for e in time_series_attrs)
            self.ts_x_views = itertools.product(self.views, self.time_series_attrs)
        else:
            self.use_time_series = False
            self.views, self.time_series_attrs = (), ()

    def _prepare_data_subset(self, data: OrchidDataModule, subset: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract and process from the data module, specifically to handle missing values and categorical attributes.

        Args:
            data: ORCHID data module.
            subset: Subset of the data to extract (e.g. "train", "test").

        Returns:
            Tuple of data extracted from the subset:
                - DataFrame of tabular data to use as input features, w/ missing values marked as np.nan.
                - Numpy array of target labels.
        """
        # Make sure the subset has been set up before extracting the data
        match subset:
            case Subset.TRAIN | Subset.VAL:
                data.setup(stage=TrainerFn.FITTING)
            case Subset.TEST:
                data.setup(stage=TrainerFn.TESTING)
            case Subset.PREDICT:
                data.setup(stage=TrainerFn.PREDICTING)
            case _:
                raise ValueError(f"Invalid subset: {subset}")

        # Select the appropriate subset of the data
        dataloader = getattr(data, f"{subset}_dataloader")()

        # For each tabular feature, save the vectors of values over each batch
        tab_data: Dict[TabularAttribute, List[np.ndarray]] = {}
        for batch, attr in itertools.product(dataloader, [*self.tabular_attrs, self.target_attr]):
            attr_batch_data = batch[attr].detach().cpu().numpy()
            tab_data.setdefault(attr, []).append(attr_batch_data)

        # Concatenate the vectors of batches of tabular features into vectors over the entire training set
        tab_data_stacked = {tab_attr: np.hstack(batch_vals) for tab_attr, batch_vals in tab_data.items()}

        # Set aside the target labels
        target = tab_data_stacked.pop(self.target_attr)

        # Create a dataframe for the training data,
        # and cast categorical attributes to the appropriate data type
        cat_attrs = [attr for attr in self.tabular_attrs if attr in TabularAttribute.categorical_attrs()]
        tab_df = pd.DataFrame(tab_data_stacked).astype({attr: "category" for attr in cat_attrs})

        # After casting categorical attributes to the appropriate data type,
        # mark missing values as `np.nan` so that they can be handled properly by the model
        tab_df[cat_attrs] = tab_df[cat_attrs].replace(MISSING_CAT_ATTR, np.nan)

        if self.use_time_series:
            assert self.time_series_attrs, "Time series attributes must be provided if use_time_series is True."
            ts_data: Dict[Tuple[str, str], List[Tensor]] = {}
            for batch in dataloader:
                ts_batch_data = filter_time_series_attributes(batch, self.views, self.time_series_attrs)[
                    0
                ]  # Do not use masked attributes
                for cross_attrs in ts_batch_data:
                    cross_attrs_column: Tuple[str, str] = (cross_attrs[0].__str__(), cross_attrs[1].__str__())
                    ts_data_value = F.interpolate(
                        np.expand_dims(ts_batch_data[cross_attrs], axis=1), size=64, mode="linear"
                    ).squeeze(dim=1)
                    ts_data.setdefault(cross_attrs_column, []).append(ts_data_value)

            # Concatenate the vectors of batches of time series features into vectors over the entire training set
            # for ts_attr, batch_vals in ts_data.items():
            #     print(f"batch vals len {len(batch_vals)}")
            #     print(f"batch vals shape{batch_vals[0].shape}")
            #     print(f"batch vals shape{batch_vals[-1].shape}")
            ts_data_stacked = {ts_attr: np.vstack(batch_vals) for ts_attr, batch_vals in ts_data.items()}
            # Create an empty list to store DataFrames for each attribute
            dfs = []

            for ts_attr, values in ts_data_stacked.items():
                df = pd.DataFrame(values, columns=[f"{ts_attr}_{i}" for i in range(64)])
                dfs.append(df)

            # Concatenate all DataFrames horizontally
            ts_df = pd.concat(dfs, axis=1)

            # Create a dataframe for the time series data
            # ts_df = pd.DataFrame(ts_data)

            # Merge the tabular and time series dataframes
            tab_df = pd.concat([tab_df, ts_df], axis=1)

        return tab_df, target

    def fit(self, data: OrchidDataModule) -> "CardiacRecordsPredictionTask":
        """Fit the model to the training set.

        Args:
            data: ORCHID data module.

        Returns:
            The fitted model.
        """
        X, y = self._prepare_data_subset(data, "train")

        # Fit the model based on sklearn's `BaseEstimator` API
        self.model = self.model.fit(X, y)

        return self

    def score(self, data: OrchidDataModule) -> Dict[str, float]:
        """Measure the model's performance on the test set.

        Args:
            data: ORCHID data module.

        Returns:
            Dictionary of model's metrics (e.g. accuracy, AUROC, etc.) on the test set.
        """
        # Extract the tabular data (i.e. inputs and target labels) from the test set
        X, y = self._prepare_data_subset(data, "test")

        # Perform inference on the test set, keeping intermediate predictions (i.e. class probabilities)
        predict_proba = self.model.predict_proba(X)
        y_hat = np.argmax(predict_proba, axis=1)

        # Compute the model's performance metrics
        scores = {
            "acc": accuracy_score(y, y_hat),
            "auroc": roc_auc_score(y, predict_proba, multi_class="ovr"),
            "auprc": average_precision_score(y, predict_proba, average="macro"),
            "f1": f1_score(y, y_hat, average="macro"),
        }

        return scores

    def save(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model to.
        """
        self.model.save_model(path)


@hydra.main(version_base=None, config_path="../config", config_name="experiment/cardinal/records-xgb")
def main(cfg: DictConfig):
    """Fit the generic model to the tabular data from the patients."""

    from pathlib import Path

    from dataprocessing.utils.logging import configure_logging
    from hydra.core.hydra_config import HydraConfig

    configure_logging(log_to_console=True, console_level=logging.INFO)

    # Set up the task and data components
    task = hydra.utils.instantiate(cfg.task)
    data = hydra.utils.instantiate(cfg.data, _recursive_=False)

    # Fit the model to the tabular data
    task.fit(data)

    # Evaluate the model's performance on the test set
    scores = task.score(data)

    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Save the model
    if task.save_model:
        # Save the model to disk
        task.save(hydra_output_dir / cfg.model_ckpt)

    # Save the model's performance
    score_df = pd.Series(scores)
    score_df.to_csv(hydra_output_dir / cfg.scores_filename, header=["value"])

    logger.info(f"Logging model and its scores: {scores} to {hydra_output_dir}")


if __name__ == "__main__":
    # Configure environment before calling hydra main function
    # Load environment variables from `.env` file if it exists
    # Load before hydra main to allow for setting environment variables with ${oc.env:ENV_NAME}
    load_dotenv()
    # Register custom Hydra resolvers
    register_omegaconf_resolvers()

    main()
