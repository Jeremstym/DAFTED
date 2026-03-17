import os
from abc import ABC
from argparse import ArgumentParser
from typing import Dict

import pytorch_lightning as pl
from dataprocessing.data.config import DataParameters, Subset
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.utilities.combined_loader import CombinedLoader

# def custom_collate(batch):
#     # Find all possible keys
#         all_keys = set().union(*(d.keys() for d in batch))
#         collated = {k: [d.get(k, None) for d in batch] for k in all_keys}
#         return collated


class VitalDataModule(pl.LightningDataModule, ABC):
    """Top-level abstract data module from which to inherit.

    Implementations of behaviors related to data handling (e.g. data preparation) are made through this class.
    """

    def __init__(
        self,
        data_params: DataParameters,
        batch_size: int,
        test_batch_size: int = 1,
        num_workers: int = os.cpu_count() - 1,
        **kwargs
    ):
        """Initializes class instance.

        References:
            - ``num_workers`` documentation, for more detail:
              https://lightning.ai/docs/pytorch/stable/advanced/speed.html#num-workers

        Args:
            data_params: Parameters related to the data necessary to initialize networks working with this dataset.
            batch_size: Size of batches.
            num_workers: Number of subprocesses to use for data loading. ``num_workers=0`` means that the data will be
                loaded in the main process.
        """
        super().__init__()
        self.data_params = data_params
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.datasets: Dict[Subset, Dataset] = {}
        self.save_hyperparameters(ignore="data_params")

    def _dataloader(self, subset: Subset, shuffle: bool = False, batch_size: int = None) -> DataLoader:
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.datasets[subset],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=bool(self.num_workers),
            # collate_fn=custom_collate,
        )

    def train_dataloader(self, get_all: bool = False, shuffle: bool = True) -> DataLoader:  # noqa: D102
        if get_all:
            batch_size = len(self.datasets[Subset.TRAIN])
            return self._dataloader(Subset.TRAIN, shuffle=False, batch_size=batch_size)
        return self._dataloader(Subset.TRAIN, shuffle=shuffle)
        # return [
        #     DataLoader(self.datasets[subset], batch_size=None, num_workers=self.num_workers, pin_memory=True)
        #     for subset in [Subset.TRAIN]
        # ]

    def val_dataloader(self, shuffle: bool = False) -> DataLoader:  # noqa: D102
        val_loader = self._dataloader(Subset.VAL, batch_size=self.test_batch_size, shuffle=shuffle)
        train_loader = self._dataloader(Subset.TRAIN, batch_size=len(self.datasets[Subset.TRAIN]))
        return CombinedLoader({"query": val_loader, "support": train_loader}, mode="max_size_cycle")
        # return self._dataloader(Subset.VAL, batch_size=self.test_batch_size)
        # return [
        #     DataLoader(self.datasets[subset], batch_size=None, num_workers=self.num_workers, pin_memory=True)
        #     for subset in [Subset.VAL]
        # ]

    def test_dataloader(self, shuffle: bool = False) -> DataLoader:  # noqa: D102
        # return self._dataloader(Subset.TEST)
        test_loader = self._dataloader(Subset.TEST, batch_size=self.test_batch_size, shuffle=shuffle)
        train_loader = self._dataloader(Subset.TRAIN, batch_size=len(self.datasets[Subset.TRAIN]))
        return CombinedLoader({"query": test_loader, "support": train_loader}, mode="max_size_cycle")
        # return CombinedLoader({
        #     "test": DataLoader(
        #         self.datasets[Subset.TEST],
        #         batch_size=self.test_batch_size,
        #         num_workers=self.num_workers,
        #         pin_memory=True,
        #     ),
        #     "train": DataLoader(
        #         self.datasets[Subset.TRAIN],
        #         batch_size=len(self.datasets[Subset.TRAIN]),
        #         num_workers=self.num_workers,
        #         pin_memory=True,
        #     ),
        #     # DataLoader(
        #     #     self.datasets[Subset.VAL],
        #     #     batch_size=len(self.datasets[Subset.VAL]),
        #     #     num_workers=self.num_workers,
        #     #     pin_memory=True,
        #     # ),
        # })

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:  # noqa: D102
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, required=True, help="Size of batches")
        parser.add_argument(
            "--num_workers",
            type=int,
            default=os.cpu_count() - 1,
            help="Number of subprocesses to use for data loading. ``workers=0`` means that the data will be loaded in "
            "the main process",
        )
        return parser
