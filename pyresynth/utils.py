"""Utility functions."""
import numpy as np


def normalize_wavdata(data: np.ndarray) -> np.ndarray:
    """Crudely normalize NumPy array to float32 in range [-1.0, 1.0].

    :param data: Input Array
    :return: Normalized Array.
    """
    if data.dtype is np.dtype('float32'):
        # Min = -1.0, Max = +1.0
        return data
    if data.dtype is np.dtype('float64'):
        data = data.astype(np.float32)
    elif data.dtype is np.dtype('uint8'):
        # Min = 0, Max = 255
        data = (data.astype(np.float32) - 128) / 128
    else:
        # Min = -2147483648, Max = +2147483647 for int32
        # Min = -32768, Max = 32767 for int16
        data = data.astype(np.float32) / np.iinfo(data.dtype).min
    return data
