import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class FigureCallback(tf.keras.callbacks.Callback):
    """[summary]

    Args:
        figure ([type]): [description]
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

        figure, axes = plt.subplots(2, 10, figsize=(15, 3))
        quantiles = [0.05, 0.95]

        for n in range(10):
            q_test = np.quantile(original[n], quantiles)
            axes[0][n].imshow(original[n], cmap="gray", vmin=q_test[0], vmax=q_test[1])

            reshape_test_image = np.reshape(reconstructed[n], (64, 64))
            q_re = np.quantile(reshape_test_image, quantiles)
            axes[1][n].imshow(
                reshape_test_image, cmap="gray", vmin=q_re[0], vmax=q_re[1]
            )

        figure.suptitle(f"Epoch {epoch}: loss={loss}")
        figure.savefig(self.path / f"{epoch}-figure.png")
