from tensorflow.keras import backend as K


class JaccLoss(object):
    def __init__(self):
        super(JaccLoss, self).__init__()
        self.smooth = 0.0000001

    def __call__(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1 - ((intersection + self.smooth) /
                    (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + self.smooth))
