#!/usr/bin/env python3
# coding: utf-8

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xml.etree.ElementTree as ET
import tensorflow as tf
from PIL import Image
from src import const
from glob import glob
import numpy as np
import imageio
import random
import os

def create_samples(generator):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i, (input_lr, input_hr) in enumerate(generator):
        imageio.imwrite(os.path.join('samples', f'hr_{i + 1}.png'), input_hr[0])
        imageio.imwrite(os.path.join('samples', f'lr_{i + 1}.png'), input_lr[0])

class RandomBBoxGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size, dim, n_channels,
            shuffle=True, state="train", augment=None, seed=None):
        self.dim = dim
        self.batch_size = batch_size // 2
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.state = state
        self.augment = augment
        self.seed = seed
        self.on_epoch_end()
        random.seed(seed)
        self.gen = ImageDataGenerator()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def augmentation_params(self):
        params = {'rescale': 1}

        if 'rescale' in self.augment:
            params['rescale'] = self.augment['rescale']

        return params 

    @staticmethod
    def read_xml(ID):
        res = []
        xml = os.path.join(os.getcwd(), 'data', 'annotations', 'train', f'{ID.split("/")[-1].split(".")[0]}.xml')
        for dim in ET.parse(xml).getroot()[-1][-1]:
            res.append(int(round(float(dim.text))))
        return res # left, top, right, bottom

    @staticmethod
    def random_bbox():
        v = [randint(0, v) for v in self.dim[0]]
        return [min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]), max(v[1], v[3])] # left, top, right, bottom 

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size * 2, *self.dim, self.n_channels))
        y = np.empty((self.batch_size * 2), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # load images
            img = Image.open(os.path.join(ID))
            X[i] = img.crop(self.read_xml(ID)).resize(self.dim[::-1])
            X[self.batch_size+i] = img.crop(self.random_bbox()).resize(self.dim[::-1])

            if self.state == "train":
                params = self.augmentation_params() # randomize on seed
                X[i,] = self.gen.apply_transform(x=X[i,], transform_parameters=params)
                X[self.batch_size+i,] = self.gen.apply_transform(x=X[self.batch_size+i,], transform_parameters=params)

        return X, y

if __name__ == '__main__':
    data_image_paths = glob(os.path.join(os.getcwd(), 'data', 'images', 'train', '*'))
    params = {'dim': const.TARGET_SIZE,
            'batch_size': const.BATCH_SIZE,
            'n_channels': 3,
            'shuffle': True,
            'augment': {'rescale': 1/255}}
    generator = RandomBBoxGenerator(data_image_paths, state='train', seed=const.SEED, **params)
    batch = generator.__getitem__(0)
