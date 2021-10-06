import sys
import os
import time

filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from utils.tracker import ADAMLearningRateTracker
from data.data_reader import DataReader, DataLoader
from modules.optimizer import Optimizer
from configs.config import params
from utils.img_utils import get_input_image_names
from modules.cloud_net import model_arch
from modules.losses import jaccard_loss

np.random.seed(42)
tf.random.set_seed(42)


def get_model(input_row, input_col, model_name='cloud'):
    if model_name == 'cloud':
        model = model_arch(input_row, input_col)
        return model
    else:
        return ValueError('Unsupported model {}'.format(model_name))


def get_dataset():
    train_data_reader = DataReader(train_img_split, train_msk_split, img_size=params['img_size'], augment=True)
    train_dataset = DataLoader(train_data_reader, params['img_size'])(batch_size=params['batch_size'])
    train_dataset.len = len(train_data_reader)

    valid_data_reader = DataReader(val_img_split, val_msk_split, img_size=params['img_size'], augment=False)
    valid_dataset = DataLoader(valid_data_reader, params['img_size'])(batch_size=params['batch_size'])
    valid_dataset.len = len(valid_data_reader)

    return train_dataset, valid_dataset


def train():
    train_dataset, valid_dataset = get_dataset()
    train_steps_per_epoch = train_dataset.len / params['batch_size']
    valid_step_per_epoch = valid_dataset.len / params['batch_size']
    warm_steps = params['warmup_epochs'] * train_steps_per_epoch

    input_row = input_col = params['img_size']
    model = get_model(input_row, input_col)

    schedule = tf.keras.experimental.CosineDecay(
        params['init_learning_rate'], warm_steps, alpha=params['warmup_alpha'])
    optim = Optimizer('adam', schedule=schedule)()
    model.compile(optimizer=optim, loss=jaccard_loss, metrics=[jaccard_loss])
    model_checkpoint = ModelCheckpoint(os.path.join(params['saved_model_dir'], 'Cloud.h5'), monitor='val_loss',
                                       save_best_only=True)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
    csv_logger = CSVLogger('training_log_{}.log'.format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
    model.fit(train_dataset, validation_data=valid_dataset, epochs=params['n_epochs'],
              steps_per_epoch=train_steps_per_epoch, validation_steps=valid_step_per_epoch, verbose=1, callbacks=[
            early_stop, csv_logger, model_checkpoint, ADAMLearningRateTracker
        ])


if __name__ == '__main__':
    GLOBAL_PATH = params['train_dataset_dir']
    TRAIN_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_training')
    TEST_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_test')
    train_patches_csv_name = 'training_patches_95-cloud_nonempty.csv'
    df_train_img = pd.read_csv(os.path.join(TRAIN_FOLDER, train_patches_csv_name))
    val_ratio = params['val_ratio']
    train_img, train_msk = get_input_image_names(df_train_img, TRAIN_FOLDER, if_train=True)
    train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(train_img, train_msk,
                                                                                      test_size=val_ratio,
                                                                                      random_state=42, shuffle=True)
    train()
