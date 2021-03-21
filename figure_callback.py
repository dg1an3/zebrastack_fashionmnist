import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer


class FigureCallback(Callback):
    """[summary]

    Args:
        path ([type]): [description]
    """

    def __init__(self, path):
        self.path = path

    def on_train_batch_end(self, batch: int, logs: dict = None):
        """[summary]

        Args:
            batch (int): [description]
            logs ([type], optional): [description]. Defaults to None.
        """
        pass

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """[summary]

        Args:
            epoch (int): [description]
            logs (dict, optional): [description]. Defaults to None.
        """
        loss = logs["loss"]
        original = logs["original"]
        reconstructed = logs["reconstructed"]

        col_count = min(original.shape[0], 10)
        figure, axes = plt.subplots(3, col_count, figsize=(15, 5))

        # these are the layers to show histograms
        layer_names = [
            # "v1_conv2d",
            # "v2_conv2d",
            # "v4_conv2d",
            "pit_conv2d",
            "cit_conv2d",
            "ait_local",
        ]
        for bins_col in range(len(layer_names)):

            # get the layer
            layer = self.model.get_layer(layer_names[bins_col])

            # set the title for the plot
            axes[0][bins_col].set_title(layer.name)

            # extract the kernels weights
            kernel_weights = [
                kernel
                for kernel in layer.trainable_variables
                if "kernel" in kernel.name
            ][0]

            axes[0][bins_col].hist(kernel_weights.numpy().reshape(-1))

        quantiles = [0.05, 0.95]
        for n in range(col_count):
            q_test = np.quantile(original[n], quantiles)
            axes[1][n].imshow(original[n], cmap="gray", vmin=q_test[0], vmax=q_test[1])

            reshape_test_image = np.reshape(reconstructed[n], (64, 64))
            q_re = np.quantile(reshape_test_image, quantiles)
            axes[2][n].imshow(
                reshape_test_image, cmap="gray", vmin=q_re[0], vmax=q_re[1]
            )

        figure.suptitle(f"Epoch {epoch}: loss={loss:0.2f}")

        # make the save path, if needed
        self.path.mkdir(parents=True, exist_ok=True)

        # save and close
        figure.savefig(self.path / f"{epoch}-figure.png")
        plt.close(figure)
