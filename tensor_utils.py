"""various utility functions
"""
from typing import Generator, Union
import numpy as np


def normalize_tuple(value: Union[int, tuple], n:int, name):
    """Transforms an integer or iterable of integers into an integer tuple.

    A copy of tensorflow.python.keras.util.

    Arguments:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.

    Returns:
      A tuple of n integers.

    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise TypeError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        if len(value_tuple) != n:
            raise ValueError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    "The `"
                    + name
                    + "` argument must be a tuple of "
                    + str(n)
                    + " integers. Received: "
                    + str(value)
                    + " "
                    "including element "
                    + str(single_value)
                    + " of type"
                    + " "
                    + str(type(single_value))
                )
        return value_tuple


def normalize_data_format(value):
    """[summary]

    Args:
        value ([type]): [description]

    Returns:
        [type]: [description]
    """
    data_format = value.lower()
    assert data_format in {"channels_first", "channels_last"}
    return data_format


def normalize_padding(value):
    """[summary]

    Args:
        value ([type]): [description]

    Returns:
        [type]: [description]
    """
    padding = value.lower()
    assert padding in {"valid", "same"}
    return padding


def convert_data_format(data_format, ndim):
    """[summary]

    Args:
        data_format ([type]): [description]
        ndim ([type]): [description]

    Returns:
        [type]: [description]
    """

    channel_from_format = {
        "channels_last": {
            3: "NWC",
            4: "NHWC",
            5: "NDHWC",
        },
        "channels_first": {
            3: "NCW",
            4: "NCHW",
            5: "NCDHW",
        },
    }
    return channel_from_format[data_format][ndim]


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    Arguments:
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full".
        stride: integer.
        dilation: dilation rate, integer.

    Returns:
        The output length (integer).
    """
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)

    assert input_length is not None
    assert padding in {"same", "valid", "full"}

    output_length_calculator = {
        "same": lambda: input_length,
        "valid": lambda: input_length - dilated_filter_size + 1,
        "full": lambda: input_length + dilated_filter_size - 1,
    }

    output_length = output_length_calculator[padding]()
    return (output_length + stride - 1) // stride


def generate_batches(
    input_data: np.ndarray, batch_size: int
) -> Generator[np.ndarray, None, None]:
    """Batches and trains using a given function

    Args:
        input_data (np.ndarray): [description]
        batch_size (int): [description]

    Yields:
        Generator[np.ndarray, None, None]: [description]
    """

    step = 0
    while (step + 1) * batch_size < input_data.shape[0]:
        input_batch = input_data[step * batch_size : (step + 1) * batch_size, ...]
        yield step, input_batch
        step += 1
