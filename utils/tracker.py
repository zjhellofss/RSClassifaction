import tensorflow as tf
from tensorflow import keras


class ADAMLearningRateTracker(keras.callbacks.Callback):
    """It prints out the last used learning rate after each epoch (useful for resuming a training)
    original code: https://github.com/keras-team/keras/issues/7874#issuecomment-329347949
    """

    def __init__(self):
        super(ADAMLearningRateTracker, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):  # works only when decay in optimizer is zero
        optimizer = self.model.optimizer
        print('\n***The Basic Learning rate in this epoch is:', optimizer._decayed_lr(tf.float32).numpy(), '***\n')
