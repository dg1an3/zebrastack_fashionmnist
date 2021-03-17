import functools
import six
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


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

    def __init__(self, directions=3, freqs=[2.0, 1.0], sz=13, **kwargs):
        """[summary]

        Args:
            name ([type], optional): [description]. Defaults to None.
            directions (int, optional): [description]. Defaults to 3.
            freqs (list, optional): [description]. Defaults to [2.0, 1.0].
            sz (int, optional): [description]. Defaults to 13.
        """
        super(GaborPowerMap2D, self).__init__(
            trainable=False, activity_regularizer=None, **kwargs
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
