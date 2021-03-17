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
from custom_layers import (
    GaborPowerMap2D,
    LogSumExpPooling2D,
)
from tensorflow.keras.models import Sequential


def create_encoder(
    sz=64,
    latent_dim=8,
    locally_connected_channels=2,
    act_func="softplus",
    gabor_powermaps=False,
    logsumexp_pool=False,
):
    """creates the encoder side of the autoencoder, for the parameters sz and latent_dim
    # Static parameters
        sz (int): sz x sz input
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
            Input(shape=(sz, sz, 1), name="retina_{}".format(sz)),
            ####
            #### V1 layers
            GaborPowerMap2D(16, (5, 5), name="v1_powmap")
            if gabor_powermaps
            else Conv2D(
                16, (5, 5), name="v1_conv2d", activation=act_func, padding="same"
            ),
            LogSumExpPooling2D(name="v1_pool")
            if logsumexp_pool
            else MaxPooling2D((2, 2), name="v1_maxpool", padding="same"),
            SpatialDropout2D(0.1, name="v1_dropout"),
            ####
            #### V2 layers
            GaborPowerMap2D(16, (3, 3), name="v2_powmap")
            if gabor_powermaps
            else Conv2D(
                16, (3, 3), name="v2_conv2d", activation=act_func, padding="same"
            ),
            LogSumExpPooling2D(name="v2_pool")
            if logsumexp_pool
            else MaxPooling2D((2, 2), name="v2_maxpool", padding="same"),
            ####
            #### V4 layers
            GaborPowerMap2D(32, (3, 3), name="v4_powmap")
            if gabor_powermaps
            else Conv2D(
                32, (3, 3), name="v4_conv2d", activation=act_func, padding="same"
            ),
            LogSumExpPooling2D(name="v4_pool")
            if logsumexp_pool
            else MaxPooling2D((2, 2), name="v4_maxpool", padding="same"),
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


class ZebraStackModel(object):
    def __init__(self):
        pass