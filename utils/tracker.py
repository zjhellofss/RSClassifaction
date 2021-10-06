from tensorflow import keras
import tensorflow.keras.backend as K


class ADAMLearningRateTracker(keras.callbacks.Callback):
    """It prints out the last used learning rate after each epoch (useful for resuming a training)
    original code: https://github.com/keras-team/keras/issues/7874#issuecomment-329347949
    """

    def __init__(self):
        super(ADAMLearningRateTracker, self).__init__()

    def on_epoch_end(self, epoch, logs={}):  # works only when decay in optimizer is zero
        optimizer = self.model.optimizer
        # t = K.cast(optimizer.iterations, K.floatx()) + 1
        # lr_t = K.eval(optimizer.lr * (K.sqrt(1. - K.pow(optimizer.beta_2, t)) /
        #                               (1. - K.pow(optimizer.beta_1, t))))
        # print('\n***The last Actual Learning rate in this epoch is:', lr_t,'***\n')
        print('\n***The last Basic Learning rate in this epoch is:', K.eval(optimizer.lr), '***\n')
