import glob
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from augmentation.augmentations import flipping_img_and_msk, rotate_cclk_img_and_msk, rotate_clk_img_and_msk, \
    zoom_img_and_msk
from utils.img_utils import get_input_image_names


class DataReader(object):
    def __init__(self, imgs, annotations, img_size=384, augment=False, test=False,
                 max_possible_input_value=65535):
        self.annotations = annotations
        self.imgs = imgs
        self.img_size = img_size  # image_target_size
        self.augment = augment
        self.idx = range(len(self.annotations))
        self.test = test
        self.max_possible_input_value = max_possible_input_value

    def __len__(self):
        return len(self.annotations)

    def iter(self):
        for i in self.idx:
            yield self[i]

    def __getitem__(self, idx):
        file = self.imgs[idx]

        image_red = imread(file[0])
        image_green = imread(file[1])
        image_blue = imread(file[2])
        image_nir = imread(file[3])

        image = np.stack((image_red, image_green, image_blue, image_nir), axis=-1)
        image = resize(image, (self.img_size, self.img_size), preserve_range=True, mode='symmetric')
        if self.test:
            return image
        else:
            mask = self.annotations[idx]
            mask = imread(mask)
            mask = resize(mask, (self.img_size, self.img_size), preserve_range=True, mode='symmetric')
            if self.augment:
                rnd_flip = np.random.randint(2, dtype=int)
                rnd_rotate_clk = np.random.randint(2, dtype=int)
                rnd_rotate_cclk = np.random.randint(2, dtype=int)
                rnd_zoom = np.random.randint(2, dtype=int)

                if rnd_flip == 1:
                    image, mask = flipping_img_and_msk(image, mask)

                if rnd_rotate_clk == 1:
                    image, mask = rotate_clk_img_and_msk(image, mask)

                if rnd_rotate_cclk == 1:
                    image, mask = rotate_cclk_img_and_msk(image, mask)

                if rnd_zoom == 1:
                    image, mask = zoom_img_and_msk(image, mask)

            mask = mask[..., np.newaxis]
            mask /= 255
            image /= self.max_possible_input_value
            return image, mask


class DataLoader(object):
    '''
    data pipeline from data_reader (image,label) to tf.data
    '''

    def __init__(self, data_reader, img_size=384):
        self.data_reader = data_reader
        self.img_size = img_size

    def __call__(self, batch_size=8, train_dataset=True):

        if train_dataset:
            dataset = tf.data.Dataset.from_generator(self.data_reader.iter,
                                                     output_types=(tf.float32, tf.float32),
                                                     output_shapes=(
                                                         [self.img_size, self.img_size, 4],
                                                         [self.img_size, self.img_size, 1]))
            dataset = dataset.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            return dataset
        else:
            dataset = tf.data.Dataset.from_generator(self.data_reader.iter,
                                                     output_types=(tf.float32),
                                                     output_shapes=(
                                                         [self.img_size, self.img_size, 4]))

            dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            return dataset
