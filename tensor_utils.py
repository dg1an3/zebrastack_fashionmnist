from tensorflow_addons.utils.keras_utils import normalize_tuple


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
