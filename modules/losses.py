from tensorflow.keras import backend as K


def jaccard_loss(y_true, y_pred):
    smooth = 0.0000001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((intersection + smooth) /
                (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))
