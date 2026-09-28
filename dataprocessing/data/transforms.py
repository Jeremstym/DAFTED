import functools

import numpy as np
import torch
import torchvision.transforms.functional as F
from dataprocessing.utils.image.transform import segmentation_to_tensor
from scipy import interpolate, signal
from torch import Tensor


class Interp1d:
    """Interpolates data points in a signal to reach a target number of data points."""

    def __init__(self, num: int, **interp1d_kwargs):
        """Initializes class instance.

        Args:
            num: Number of samples to interpolate from the original signal.
            **interp1d_kwargs: Additional parameters to pass along to ``scipy.signal.resample``.
        """
        super().__init__()
        self.interp_x_coords = np.linspace(0, 1, num=num)
        self.interp1d_kwargs = interp1d_kwargs

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Interpolates input signal.

        Args:
            signal: (M), Signal to interpolate.

        Returns:
            (N), Interpolated signal.
        """
        signal_x_coords = np.linspace(0, 1, num=len(signal))
        f = interpolate.interp1d(signal_x_coords, signal, **self.interp1d_kwargs)
        return f(self.interp_x_coords)
