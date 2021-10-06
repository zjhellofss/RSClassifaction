import tensorflow as tf


class Optimizer(object):
    def __init__(self, optimizer_method, schedule):
        self.optimizer_method = optimizer_method
        self.schedule = schedule

    def __call__(self):
        if self.optimizer_method == 'adam':
            return tf.keras.optimizers.Adam(self.schedule)
        elif self.optimizer_method == 'rmsprop':
            return tf.keras.optimizers.RMSprop(self.schedule)
        elif self.optimizer_method == 'sgd':
            return tf.keras.optimizers.SGD(self.schedule)
        else:
            raise ValueError('Unsupported optimizer {}'.format(self.optimizer_method))
