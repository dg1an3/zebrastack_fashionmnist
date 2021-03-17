import functools
import six
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Pooling2D


def kernels2tensor(kernels: List[np.ndarray], dtype=tf.float32) -> tf.Tensor:
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


def make_gabor_kernels(xs, ys, directions=3, freqs=[2.0, 1.0]) -> tf.Tensor:
    """makes a bank of gabor kernels as a complex tensor

    Args:
        xs ([type]): [description]
        ys ([type]): [description]
        directions (int, optional): [description]. Defaults to 3.
        freqs (list, optional): [description]. Defaults to [2.0, 1.0].

    Returns:
        tf.Tensor: complex tensor with a kernel on each channel
    """
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


class GaborPowerMap2D(Layer):
    """non-trainable gabor filter bank layer"""

    def __init__(self, name=None, directions=3, freqs=[2.0, 1.0], sz=13, **kwargs):
        """[summary]

        Args:
            name ([type], optional): [description]. Defaults to None.
            directions (int, optional): [description]. Defaults to 3.
            freqs (list, optional): [description]. Defaults to [2.0, 1.0].
            sz (int, optional): [description]. Defaults to 13.
        """
        super(GaborPowerMap2D, self).__init__(
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


def logsumexp_pool(
    value,
    ksize,
    strides=[1, 2, 2, 1],
    padding="SAME",
    data_format="NHWC",
    scale_up=1e2,
    name=None,
):
    """function to calculate the log(sum(exp(x))) function,
    which is a continuous approximation to the max function
    """
    assert len(value.shape) == 4
    value = tf.transpose(value, perm=[3, 1, 2, 0])
    scaled = tf.scalar_mul(scale_up, value)
    patched_response = tf.image.extract_patches(
        scaled,
        sizes=[1, 2, 2, 1],  # TODO: pass these in???
        strides=strides,
        rates=[1, 1, 1, 1],
        padding=padding,
    )
    logsumexp_result = tf.math.reduce_logsumexp(patched_response, axis=-1)
    descaled = tf.scalar_mul(1.0 / scale_up, logsumexp_result)
    return tf.transpose(tf.expand_dims(descaled, axis=-1), perm=[3, 1, 2, 0])


class LogSumExpPooling2D(Pooling2D):
    """[summary]"""

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        scale_up=1e2,
        **kwargs
    ):
        """[summary]

        Args:
            pool_size (tuple, optional): [description]. Defaults to (2, 2).
            strides ([type], optional): [description]. Defaults to None.
            padding (str, optional): [description]. Defaults to "valid".
            data_format ([type], optional): [description]. Defaults to None.
            scale_up ([type], optional): [description]. Defaults to 1e2.
        """
        super(LogSumExpPooling2D, self).__init__(
            logsumexp_pool,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


if __name__ == "__main__":
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
