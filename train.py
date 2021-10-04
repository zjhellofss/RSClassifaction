import sys
import os

filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import time
import shutil
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from data.data_reader import DataReader, DataLoader
from modules.optimizer import Optimizer, LrScheduler
from configs.config import params
from utils.img_utils import get_input_image_names
from modules.cloud_net import model_arch
from modules.losses import JaccLoss

np.random.seed(42)
tf.random.set_seed(42)


class Trainer(object):
    """ Trainer class that uses the dataset and model to train
    # Usage
    data_loader = tf.data.Dataset()
    trainer = Trainer(params)
    trainer.train(data_loader)
    """

    def __init__(self, params):
        """ Constructor
        :param params: dict, with dir and training parameters
        """
        self.params = params
        if os.path.exists(self.params['log_dir']):
            shutil.rmtree(self.params['log_dir'])
        self.log_writer = tf.summary.create_file_writer(self.params['log_dir'])
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.build_model()

    def build_model(self):
        """ Build the model,
        define the training strategy and model, loss, optimizer
        :return:
        """
        if self.params['multi_gpus']:
            self.strategy = tf.distribute.MirroredStrategy(devices=None)
        else:
            self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

        with self.strategy.scope():
            self.model = model_arch(params['img_size'], params['img_size'], 4, 1)
            self.loss_fn = JaccLoss()
            self.optimizer = Optimizer('adam')()

    def train(self, train_dataset, valid_dataset=None):
        """ train function
        :param train_dataset: train dataset built by tf.data
        :param valid_dataset: valid dataset build by td.data, optional
        :param transfer: pretrain
        :return:
        """
        valid_min_loss = 1e8
        steps_per_epoch = train_dataset.len / self.params['batch_size']
        self.total_steps = int(self.params['n_epochs'] * steps_per_epoch)
        self.params['warmup_steps'] = self.params['warmup_epochs'] * steps_per_epoch

        with self.strategy.scope():
            self.lr_scheduler = LrScheduler(self.total_steps, self.params, scheduler_method='cosine')

            print("Train from scratch")
            self.model.summary()

        train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)

        for epoch in range(1, self.params['n_epochs'] + 1):
            start_time = time.time()
            train_loss = []
            for step, (image, target) in enumerate(train_dataset):
                loss = self.dist_train_step(image, target)
                train_loss.append(loss.numpy())
                train_loss_np = np.asarray(train_loss, dtype=np.float32)
                if step % 100 == 0:
                    print('=> Epoch {}, Step {},Step Loss {:.5f} Total loss {:.5f}'.format(epoch,
                                                                                           self.global_step.numpy(),
                                                                                           loss.numpy(),
                                                                                           np.mean(train_loss_np)))
                with self.log_writer.as_default():
                    tf.summary.scalar('loss', loss, step=self.global_step)
                    tf.summary.scalar('lr', self.optimizer.lr, step=self.global_step)
                self.log_writer.flush()

            valid_loss = self.validate(valid_dataset)
            print('=> Epoch {}, Total validation loss {:.5f}'.format(epoch, valid_loss))
            end_time = time.time()
            duration = int(end_time) - int(start_time)
            print('=> Epoch {}, Total duration time {}'.format(epoch, duration))
            if valid_loss < valid_min_loss:
                valid_min_loss = valid_loss
                self.export_model()
                print('Saving model path for epoch {}'.format(epoch))

    # @tf.function
    def train_step(self, image, target):
        with tf.GradientTape() as tape:
            logit = self.model(image)
            total_loss = self.loss_fn(target, logit)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        lr = self.lr_scheduler.step()
        self.optimizer.lr.assign(lr)
        self.global_step.assign_add(1)
        return total_loss

    @tf.function
    def dist_train_step(self, image, target):
        with self.strategy.scope():
            loss = self.strategy.run(self.train_step, args=(image, target))
            total_loss_mean = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
            return total_loss_mean

    def validate(self, valid_dataset):
        valid_loss = []
        for step, (image, target) in enumerate(valid_dataset):
            step_valid_loss = self.valid_step(image, target)
            valid_loss.append(step_valid_loss)
        return np.mean(valid_loss)

    def valid_step(self, image, label):
        logit = self.model(image)
        total_loss = self.loss_fn(label, logit)
        return total_loss

    def export_model(self):
        self.model.save(os.path.join(self.params['saved_model_dir'], 'Cloud.h5'))
        print("pb model saved in {}".format(self.params['saved_model_dir']))


def train():
    train_data_reader = DataReader(train_img_split, train_msk_split, img_size=params['img_size'], augment=True)
    train_dataset = DataLoader(train_data_reader, params['img_size'])(batch_size=params['batch_size'])
    train_dataset.len = len(train_data_reader)

    valid_data_reader = DataReader(val_img_split, val_msk_split, img_size=params['img_size'], augment=False)
    valid_dataset = DataLoader(valid_data_reader, params['img_size'])(batch_size=params['batch_size'])
    valid_dataset.len = len(valid_data_reader)

    trainer.train(train_dataset, valid_dataset)


if __name__ == '__main__':
    trainer = Trainer(params)
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
