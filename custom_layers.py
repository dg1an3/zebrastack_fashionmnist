import functools
import six

import numpy as np
import tensorflow as tf
from tensorflow import TensorShape
from tensorflow.keras import regularizers, initializers, activations, constraints
from tensorflow.keras.layers import Layer, InputSpec, Pooling2D
from tensorflow.keras.layers import normalize_padding, normalize_data_format
from tensorflow.keras.utils import (
    normalize_tuple,
    convert_data_format,
    conv_output_length,
)
import tensorflow.keras.backend as K


def squeeze_batch_dims(inp: tf.Tensor, op, inner_rank):
    shape = inp.shape

    inner_shape = shape[-inner_rank:]
    batch_shape = shape[:-inner_rank]

    assert isinstance(inner_shape, TensorShape)
    inp_reshaped = tf.reshape(inp, [-1] + inner_shape.as_list())

    out_reshaped = op(inp_reshaped)

    out_inner_shape = out_reshaped.shape[-inner_rank:]

    out = tf.reshape(out_reshaped, batch_shape + out_inner_shape)

    out.set_shape(inp.shape[:-inner_rank] + out.shape[-inner_rank:])
    return out


def logsumexp_pooling(t: tf.Tensor, scale_up=1e2):
    """"""
    assert len(t.shape) == 4
    t = tf.transpose(t, perm=[3, 1, 2, 0])
    scaled = tf.scalar_mul(scale_up, t)
    patched_response = tf.image.extract_patches(
        scaled,
        sizes=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        rates=[1, 1, 1, 1],
        padding="SAME",
    )
    logsumexp_result = tf.math.reduce_logsumexp(patched_response, axis=-1)
    descaled = tf.scalar_mul(1.0 / scale_up, logsumexp_result)
    return tf.transpose(tf.expand_dims(descaled, axis=-1), perm=[3, 1, 2, 0])


def kernels2tensor(kernels, dtype=tf.float32):
    kernels = np.moveaxis(np.expand_dims(kernels, axis=-1), 0, -1)
    return tf.constant(kernels, dtype=dtype)


def complex_exp(xs, ys, freq, angle_rad):
    return np.exp(freq * (xs * np.sin(angle_rad) + ys * np.cos(angle_rad)) * 1.0j)


def gauss(xs, ys, sigma):
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -(xs * xs + ys * ys) / (2.0 * sigma * sigma)
    )


def make_meshgrid(sz=9):
    return np.meshgrid(
        np.linspace(-(sz // 2), sz // 2, sz), np.linspace(-(sz // 2), sz // 2, sz)
    )


def make_gabor_kernels(xs, ys, directions=3, freqs=[2.0, 1.0]):
    """ """
    angles_rad = [n * np.pi / float(directions) for n in range(directions)]
    sine_kernels = kernels2tensor(
        [
            complex_exp(xs, ys, freq, angle_rad)
            for freq in freqs
            for angle_rad in angles_rad
        ]
    )
    sigmas = [2.0 / freq for freq in freqs]
    gauss_kernels = kernels2tensor([gauss(xs, ys, sigma) for sigma in sigmas])
    gauss_kernels = np.repeat(
        gauss_kernels, sine_kernels.shape[-1] // gauss_kernels.shape[-1], axis=-1
    )

    bank = gauss_kernels * sine_kernels
    g0 = kernels2tensor([gauss(xs, ys, 4.0 / freqs[-1])])
    return tf.concat([bank, g0], -1)


def make_test_kernels():
    import matplotlib.pyplot as plt

    directions = 5
    freqs = [2.0, 1.0]
    xs, ys = make_meshgrid(sz=9)
    gabor_kernels = make_gabor_kernels(xs, ys, directions=directions, freqs=freqs)

    _, axs = plt.subplots(
        len(freqs) + 1, directions, figsize=(directions * 3, len(freqs) * 3)
    )
    for n in range(directions):
        for m in range(len(freqs)):
            img = tf.squeeze(gabor_kernels[..., 0, m * directions + n])
            axs[m][n].imshow(tf.math.real(img), cmap="plasma")
    g0 = tf.squeeze(gabor_kernels[..., 0, -1])
    for n in range(directions):
        axs[len(freqs)][n].imshow(tf.squeeze(g0))


class GaborPowerMapLayer(Layer):
    """non-trainable gabor filter bank layer"""

    def __init__(self, name=None, directions=3, freqs=[2.0, 1.0], sz=13, **kwargs):
        super(GaborPowerMapLayer, self).__init__(
            trainable=False, name=name, activity_regularizer=None, **kwargs
        )
        self.directions = directions
        self.freqs = freqs
        self.sz = sz

    def build(self, input_shape):
        # computer gabor filter bank
        xs, ys = make_meshgrid(sz=self.sz)
        kernels = make_gabor_kernels(
            xs, ys, directions=self.directions, freqs=self.freqs
        )
        self._real_kernels = tf.math.real(kernels)
        self._imag_kernels = tf.math.imag(kernels)

    def call(self, inputs):
        response = (
            tf.nn.conv2d(inputs, self._real_kernels, strides=1, padding="SAME") ** 2
            + tf.nn.conv2d(inputs, self._imag_kernels, strides=1, padding="SAME") ** 2
        )
        return response


class LogSumExpPooling2D(Pooling2D):
    """[summary]

    Args:
        Pooling2D ([type]): [description]
    """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs
    ):
        super(LogSumExpPooling2D, self).__init__(
            logsumexp_pooling,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


class DummyConv2D(Layer):
    """dummy implementation of trainable Conv2D, for comparison

    Args:
        Layer ([type]): [description]
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        conv_op=None,
        **kwargs
    ):
        super(DummyConv2D, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs
        )
        self.rank = 2

        if isinstance(filters, float):
            filters = int(filters)
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = normalize_tuple(kernel_size, rank, "kernel_size")
        self.strides = normalize_tuple(strides, rank, "strides")
        self.padding = normalize_padding(padding)
        self.data_format = normalize_data_format(data_format)
        self.dilation_rate = normalize_tuple(dilation_rate, rank, "dilation_rate")

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)

        self._validate_init()
        self._channels_first = self.data_format == "channels_first"
        self._tf_data_format = convert_data_format(self.data_format, self.rank + 2)

    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the number of "
                "groups. Received: groups={}, filters={}".format(
                    self.groups, self.filters
                )
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0(s). "
                "Received: %s" % (self.kernel_size,)
            )

    def build(self, input_shape):
        input_shape = TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                "The number of input channels must be evenly divisible by the number "
                "of groups. Received groups={}, but the input has {} channels "
                "(full input shape is {}).".format(
                    self.groups, input_channel, input_shape
                )
            )

        kernel_shape = self.kernel_size + (
            input_channel // self.groups,
            self.filters,
        )

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None

        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )

        # Convert Keras formats to TF native formats.
        if isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__

        self._convolution_op = functools.partial(
            tf.nn.conv2d,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name,
        )
        self.built = True

    def call(self, inputs):
        outputs = self._convolution_op(inputs, self.kernel)

        if self.use_bias:
            output_rank = outputs.shape.rank

            # Handle multiple batch dimensions.
            if output_rank is not None and output_rank > 2 + self.rank:

                outputs = squeeze_batch_dims(
                    outputs,
                    lambda o: K.bias_add(
                        o, self.bias, data_format=self._tf_data_format
                    ),
                    inner_rank=self.rank + 1,
                )
        else:
            outputs = K.bias_add(outputs, self.bias, data_format=self._tf_data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        return [
            conv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i],
            )
            for i, length in enumerate(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == "channels_last":
            return TensorShape(
                input_shape[:batch_rank]
                + self._spatial_output_shape(input_shape[batch_rank:-1])
                + [self.filters]
            )

        else:
            return TensorShape(
                input_shape[:batch_rank]
                + [self.filters]
                + self._spatial_output_shape(input_shape[batch_rank + 1 :])
            )

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super(DummyConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _get_channel_axis(self):
        if self.data_format == "channels_first":
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding
