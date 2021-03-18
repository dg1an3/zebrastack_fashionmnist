"""implements the GaborPowerMap2D keras layer
"""
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


def kernels2tensor(kernels: List[np.ndarray], dtype=tf.float32) -> tf.Tensor:
    """turns list of numpy arrays to a tensor of given typeS

    Args:
        kernels (List[np.ndarray]): input kernels
        dtype ([type], optional): resulting tensor dtype. Defaults to tf.float32.

    Returns:
        tf.Tensor: the tensor formed from the kernels.  axis for kernels is last
    """
    kernels = np.moveaxis(np.expand_dims(kernels, axis=-1), 0, -1)
    return tf.constant(kernels, dtype=dtype)


def complex_exp(x_grid, y_grid, freq, angle_rad):
    """[summary]

    Args:
        x_grid ([type]): [description]
        y_grid ([type]): [description]
        freq ([type]): [description]
        angle_rad ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.exp(
        freq * (x_grid * np.sin(angle_rad) + y_grid * np.cos(angle_rad)) * 1.0j
    )


def gauss(x_grid, y_grid, sigma):
    """[summary]

    Args:
        xs ([type]): [description]
        ys ([type]): [description]
        sigma ([type]): [description]

    Returns:
        [type]: [description]
    """
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -(x_grid * x_grid + y_grid * y_grid) / (2.0 * sigma * sigma)
    )


def make_meshgrid(size=9):
    """[summary]

    Args:
        sz (int, optional): [description]. Defaults to 9.

    Returns:
        [type]: [description]
    """
    return np.meshgrid(
        np.linspace(-(size // 2), size // 2, size),
        np.linspace(-(size // 2), size // 2, size),
    )


def make_gabor_kernels(x_grid, y_grid, directions=3, freqs=[2.0, 1.0]) -> tf.Tensor:
    """makes a bank of gabor kernels as a complex tensor

    Args:
        x_grid ([type]): [description]
        y_grid ([type]): [description]
        directions (int, optional): [description]. Defaults to 3.
        freqs (list, optional): [description]. Defaults to [2.0, 1.0].

    Returns:
        tf.Tensor: complex tensor with a kernel on each channel
    """
    angles_rad = [n * np.pi / float(directions) for n in range(directions)]
    sine_kernels = kernels2tensor(
        [
            complex_exp(x_grid, y_grid, freq, angle_rad)
            for freq in freqs
            for angle_rad in angles_rad
        ]
    )
    sigmas = [2.0 / freq for freq in freqs]
    gauss_kernels = kernels2tensor([gauss(x_grid, y_grid, sigma) for sigma in sigmas])
    gauss_kernels = np.repeat(
        gauss_kernels, sine_kernels.shape[-1] // gauss_kernels.shape[-1], axis=-1
    )

    bank = gauss_kernels * sine_kernels
    g0 = kernels2tensor([gauss(x_grid, y_grid, 4.0 / freqs[-1])])
    return tf.concat([bank, g0], -1)


class GaborPowerMap2D(Layer):
    """[summary]"""

    def __init__(self, directions=3, freqs=[2.0, 1.0], sz=13, **kwargs):

        super(GaborPowerMap2D, self).__init__(
            trainable=False, activity_regularizer=None, **kwargs
        )
        self.directions = directions
        self.freqs = freqs
        self.sz = sz

    def build(self, input_shape):
        """[summary]

        Args:
            input_shape ([type]): [description]
        """
        # computer gabor filter bank
        xs, ys = make_meshgrid(size=self.sz)
        kernels = make_gabor_kernels(
            xs, ys, directions=self.directions, freqs=self.freqs
        )
        self._real_kernels = tf.math.real(kernels)
        self._imag_kernels = tf.math.imag(kernels)

    def call(self, inputs):
        """[summary]

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """
        response = (
            tf.nn.conv2d(inputs, self._real_kernels, strides=1, padding="SAME") ** 2
            + tf.nn.conv2d(inputs, self._imag_kernels, strides=1, padding="SAME") ** 2
        )
        return response


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    directions = 5
    freqs = [2.0, 1.0]
    _x_grid, _y_grid = make_meshgrid(size=9)
    gabor_kernels = make_gabor_kernels(
        _x_grid, _y_grid, directions=directions, freqs=freqs
    )

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
