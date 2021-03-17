"""zebrastack_model_v2

Returns:
    [type]: [description]

Yields:
    [type]: [description]
"""
from datetime import datetime
import logging
from time import time
from pathlib import Path
from typing import Callable, Generator

from autologging import logged
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    SpatialDropout2D,
    ZeroPadding2D,
    LocallyConnected2D,
    Dense,
    Flatten,
    ActivityRegularization,
    Reshape,
)
from tensorflow.keras.models import Sequential

from gabor_powermap_2d import GaborPowerMap2D
from logsumexp_pooling_2d import LogSumExpPooling2D


@logged
def prepare_images(img_array: np.ndarray, size=64):
    """[summary]

    Args:
        img_array (np.ndarray): [description]
        sz (int, optional): [description]. Defaults to 64.

    Returns:
        [type]: [description]
    """
    img_array = (img_array / 255.0).astype(np.float32)
    # prepare_images._log.info(f"img_array: {img_array.shape} {img_array.dtype}")
    img_array = [resize(img_array[n], (size, size)) for n in range(img_array.shape[0])]
    img_array = np.reshape(img_array, (len(img_array), size, size, 1))
    # prepare_images._log.info(f"img_array: {img_array.shape} {img_array.dtype}")
    return img_array


def create_encoder_v1(
    size=64,
    latent_dim=8,
    locally_connected_channels=2,
    act_func="softplus",
):
    """creates the encoder side of the autoencoder, for the parameters sz and latent_dim
    # Static parameters
        size (int): size x size input
        latent_dim (int): gaussian dimensions
        locally_connected_channels = 2
    # Arguments
        <none>
    # Returns
        retina: the input layer
        encoder: the encoder model
        shape: shape of last input layer
        [z_mean, z_log_var, z]: tensors for latent space
    """

    return Sequential(
        [
            Input(shape=(size, size, 1), name="retina_{}".format(size)),
            ####
            #### V1 layers
            Conv2D(16, (5, 5), name="v1_conv2d", activation=act_func, padding="same"),
            MaxPooling2D((2, 2), name="v1_maxpool", padding="same"),
            SpatialDropout2D(0.1, name="v1_dropout"),
            ####
            #### V2 layers
            Conv2D(16, (3, 3), name="v2_conv2d", activation=act_func, padding="same"),
            MaxPooling2D((2, 2), name="v2_maxpool", padding="same"),
            ####
            #### V4 layers
            Conv2D(32, (3, 3), name="v4_conv2d", activation=act_func, padding="same"),
            MaxPooling2D((2, 2), name="v4_maxpool", padding="same"),
            ####
            #### IT Layers
            Conv2D(32, (3, 3), name="pit_conv2d", activation=act_func, padding="same"),
            Conv2D(64, (3, 3), name="cit_conv2d", activation=act_func, padding="same"),
            LocallyConnected2D(
                locally_connected_channels,
                (3, 3),
                name="ait_local",
                activation=act_func,
            ),
            ActivityRegularization(l1=0.0e-4, l2=0.0e-4, name="ait_regular"),
            ####
            #### Pulvinar
            # generate latent vector Q(z|X)
            Flatten(name="pulvinar_flatten"),
            Dense(latent_dim, name="pulvinar_dense", activation=act_func),
            Dense(latent_dim + latent_dim, name="z_mean_log_var"),
        ],
        name="v1_to_pulvinar_encoder",
    )


def create_encoder_v2(
    size=64,
    latent_dim=8,
    locally_connected_channels=2,
    act_func="softplus",
):
    """creates the encoder side of the autoencoder, for the parameters sz and latent_dim
    # Static parameters
        size (int): size x size input
        latent_dim (int): gaussian dimensions
        locally_connected_channels = 2
    # Arguments
        <none>
    # Returns
        retina: the input layer
        encoder: the encoder model
        shape: shape of last input layer
        [z_mean, z_log_var, z]: tensors for latent space
    """

    return Sequential(
        [
            Input(shape=(size, size, 1), name="retina_{}".format(size)),
            ####
            #### V1 layers
            GaborPowerMap2D(16, (5, 5), name="v1_powmap"),
            LogSumExpPooling2D(name="v1_pool"),
            SpatialDropout2D(0.1, name="v1_dropout"),
            ####
            #### V2 layers
            GaborPowerMap2D(16, (3, 3), name="v2_powmap"),
            LogSumExpPooling2D(name="v2_pool"),
            ####
            #### V4 layers
            GaborPowerMap2D(32, (3, 3), name="v4_powmap"),
            LogSumExpPooling2D(name="v4_pool"),
            ####
            #### IT Layers
            Conv2D(32, (3, 3), name="pit_conv2d", activation=act_func, padding="same"),
            Conv2D(64, (3, 3), name="cit_conv2d", activation=act_func, padding="same"),
            LocallyConnected2D(
                locally_connected_channels,
                (3, 3),
                name="ait_local",
                activation=act_func,
            ),
            ActivityRegularization(l1=0.0e-4, l2=0.0e-4, name="ait_regular"),
            ####
            #### Pulvinar
            # generate latent vector Q(z|X)
            Flatten(name="pulvinar_flatten"),
            Dense(latent_dim, name="pulvinar_dense", activation=act_func),
            Dense(latent_dim + latent_dim, name="z_mean_log_var"),
        ],
        name="v1_to_pulvinar_encoder",
    )


def create_decoder(
    dense_shape, latent_dim=8, locally_connected_channels=2, act_func="softplus"
):
    """creates the decoder side of the autoencoder, given the input shape
    Args:
        dense_shape (tuple): shape to be used for dense layer
        latent_dim (int, optional):
            latent dimension of input. Defaults to 8.
        locally_connected_channels (int, optional):
            number of channels for locally connected layer. Defaults to 2.
        act_func (str, optional):
            activation function to be used. Defaults to 'softplus'.

    Returns:
        Sequential: keras model for decoder
    """
    return Sequential(
        [
            Input(shape=(latent_dim,), name="z_sampling"),
            Dense(
                dense_shape[1] * dense_shape[2] * dense_shape[3],
                name="pulvinar_dense_back",
                activation=act_func,
            ),
            Reshape(
                (dense_shape[1], dense_shape[2], dense_shape[3]),
                name="pulvinar_antiflatten",
            ),
            ####
            #### IT retro Layers
            ZeroPadding2D(padding=(1, 1), name="ait_padding_back"),
            LocallyConnected2D(
                locally_connected_channels,
                (3, 3),
                name="ait_local_back",
                activation=act_func,
            ),
            ZeroPadding2D(padding=(1, 1), name="cit_padding_back"),
            Conv2DTranspose(
                64, (3, 3), name="cit_conv2d_trans", activation=act_func, padding="same"
            ),
            Conv2DTranspose(
                32, (3, 3), name="pit_conv2d_trans", activation=act_func, padding="same"
            ),
            ####
            #### V4 retro layers
            Conv2DTranspose(
                32, (3, 3), name="v4_conv2d_trans", activation=act_func, padding="same"
            ),
            UpSampling2D((2, 2), name="v4_upsample_back"),
            ####
            #### V2 retro layers
            Conv2DTranspose(
                16, (3, 3), name="v2_conv2d_trans", activation=act_func, padding="same"
            ),
            UpSampling2D((2, 2), name="v2_upsample_back"),
            ####
            #### V1 retro layers
            Conv2D(
                1,
                (5, 5),
                name="v1_conv2d_5x5_back",
                # activation='sigmoid', no sigmoid == return logits
                padding="same",
            ),
            UpSampling2D((2, 2), name="v1_upsample_back"),
        ],
        name="pulvinar_to_v1_decoder",
    )


# @traced
def log_normal_pdf(sample, mean, logvar, raxis=1):
    """[summary]

    Args:
        sample ([type]): [description]
        mean ([type]): [description]
        logvar ([type]): [description]
        raxis (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


# @traced
def reparameterize(mean, logvar):
    """[summary]

    Args:
        mean ([type]): [description]
        logvar ([type]): [description]

    Returns:
        [type]: [description]
    """
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean


# @traced
def compute_loss(encoder: Sequential, decoder: Sequential, value: tf.Tensor):
    """compute the VAE loss function

    Args:
        encoder ([type]): [description]
        decoder ([type]): [description]
        value ([type]): [description]

    Returns:
        [type]: [description]
    """
    encoded_value = encoder(value)
    mean, logvar = tf.split(encoded_value, 2, 1)
    z_value = reparameterize(mean, logvar)
    x_logit = decoder(z_value)
    # tf.print(f"x_logit, x.shape = {x_logit}, {x.shape}")
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=value)
    # tf.print(f"cross_ent.shape = {cross_ent.shape}")
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z_value, 0.0, 0.0)
    logqz_x = log_normal_pdf(z_value, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(encoder, decoder, x_samples, optimizer):
    """Executes one training step and returns the loss.
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    all_trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    with tf.GradientTape() as tape:
        loss = compute_loss(encoder, decoder, x_samples)
        # tf.print(f"loss value = {loss}")
    gradients = tape.gradient(loss, all_trainable_variables)
    optimizer.apply_gradients(zip(gradients, all_trainable_variables))


def generate_batches(
    input_data: np.ndarray, batch_size: int
) -> Generator[np.ndarray, None, None]:
    """Batches and trains using a given function"""
    step = 0
    while (step + 1) * batch_size < input_data.shape[0]:
        input_batch = input_data[step * batch_size : (step + 1) * batch_size, ...]
        yield step, input_batch
        step += 1


def forall_batch(
    input_batches: Generator[np.ndarray, None, None],
    train_func: Callable[[np.ndarray], None],
    tb_callback=None,
):
    """Batches and trains using a given function"""
    start_time = time()
    logs = {"loss": None, "mean_absolute_error": None, "output": None}
    for step, input_batch in input_batches:
        if not step % 10000:
            logging.info(f"{step}: {input_batch.shape}")

        if tb_callback:
            tb_callback.on_train_batch_begin(step, logs=logs)

        train_func(input_batch)

        if tb_callback:
            tb_callback.on_train_batch_end(step, logs=logs)
    end_time = time()
    return end_time - start_time


class ZebraStackModel:
    """[summary]"""

    def __init__(self, latent_dim=8):
        self.encoder = create_encoder_v1(latent_dim=latent_dim)
        dense_shape = self.encoder.get_layer("ait_local").output_shape
        self.decoder = create_decoder(dense_shape, latent_dim=latent_dim)

    def train(self, train_images, test_images):
        """[summary]

        Args:
            train_images ([type]): [description]
            test_images ([type]): [description]
        """
        log_dir = Path(".") / "logs" / "fit" / datetime.now().strftime("%Y%m%d-%H%M%S")
        logging.info(log_dir)

        tb_callback = (
            None  # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        )
        if tb_callback:
            tb_callback.set_model(self.encoder)
            # logs = {"loss": None, "mean_absolute_error": None, "output": None}
            tb_callback.on_train_begin()

        batch_size = 16
        optimizer = tf.keras.optimizers.Adam(1e-4)
        loss = tf.keras.metrics.Mean()
        for epoch in range(1, 121):
            if tb_callback:
                tb_callback.on_epoch_begin(epoch)

            # perform training
            train_batches = generate_batches(train_images, batch_size)
            time_elapsed = forall_batch(
                train_batches,
                lambda batch: train_step(self.encoder, self.decoder, batch, optimizer),
            )

            # calculate test results
            test_batches = generate_batches(test_images, 16)
            loss.reset_states()
            forall_batch(
                test_batches,
                lambda batch: loss(compute_loss(self.encoder, self.decoder, batch)),
            )
            elbo = -loss.result().numpy()
            logging.info(
                f"Epoch: {epoch}, test ELBO: {elbo}, time elapsed: {time_elapsed}"
            )

            if tb_callback:
                tb_callback.on_epoch_end(epoch)

        if tb_callback:
            tb_callback.on_train_end()

    def recognize(self, image):
        """[summary]

        Args:
            image ([type]): [description]
        """
        return self.encoder(image)

    def generate(self, latent):
        """[summary]

        Args:
            latent ([type]): [description]
        """
        return self.decoder(latent)


if __name__ == "__main__":
    model = ZebraStackModel(latent_dim=8)
    model.encoder.summary()
    model.decoder.summary()

    (_train_images, _train_labels), (
        _test_images,
        _test_labels,
    ) = tf.keras.datasets.fashion_mnist.load_data()
    _train_images = _train_images[: len(_train_images) // 1]
    _test_images = _test_images[: len(_test_images) // 1]
    logging.info(f"train_images: {_train_images.shape} {_train_images.dtype}")

    # now do training
    model.train(_train_images, _test_images)

    # model.recognize(_test_images)
    # model.generate(latent)