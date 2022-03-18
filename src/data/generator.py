#!/usr/bin/env python3
# coding: utf-8

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xml.etree.ElementTree as ET
import tensorflow as tf
from PIL import Image
from src import const
from glob import glob
from pathlib import Path
import numpy as np
import imageio
import random
import os

def create_samples(generator):
    Path('samples').mkdir(parents=True, exist_ok=True)
    X, y = generator.__getitem__(0)
    for i in range(const.BATCH_SIZE):
        imageio.imwrite(os.path.join('samples', f'{i + 1}_{y[i]}.png'), X[i])

def get_image_paths():
    paths = []
    for directory in ['train', 'val']:
        paths.append(list(set(map(lambda x: x.split('/')[-1].split('.')[0], glob(os.path.join(os.getcwd(), 'data', 'images', directory, '*')))) & 
            set(map(lambda x: x.split('/')[-1].split('.')[0], glob(os.path.join(os.getcwd(), 'data', 'annotations', directory, '*'))))))

    return paths 

class RandomBBoxGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size, dim, n_channels, base_dir = '.',
            shuffle=True, state="train", augment=None, seed=None):
        self.dim = dim
        self.batch_size = batch_size // 2
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.base_dir = base_dir
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
    def random_bbox(bounds):
        v = [random.randint(0, v) for v in bounds]
        return [min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]), max(v[1], v[3])] # left, top, right, bottom 

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size * 2, *self.dim, self.n_channels))
        y = np.empty((self.batch_size * 2), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ID = os.path.join(self.base_dir, 'images', self.state, f'{ID}.png')

            img = Image.open(ID)
            X[i] = img.crop(self.read_xml(ID)).resize(self.dim[::-1])
            X[self.batch_size+i] = img.crop(self.random_bbox(img.getbbox())).resize(self.dim[::-1])

            y[i] = 1
            y[self.batch_size+i] = 0

            if self.state == "train":
                params = self.augmentation_params() # randomize on seed
                X[i,] = self.gen.apply_transform(x=X[i,], transform_parameters=params)
                X[self.batch_size+i,] = self.gen.apply_transform(x=X[self.batch_size+i,], transform_parameters=params)

        order = np.arange(self.batch_size * 2)
        np.random.shuffle(order)

        return X[order], y[order]

if __name__ == '__main__':
    train_image_paths, val_image_paths = get_image_paths()
    params = {'base_dir': os.path.join(os.getcwd(), 'data'),
            'dim': const.TARGET_SIZE,
            'batch_size': const.BATCH_SIZE,
            'n_channels': 3,
            'shuffle': True,
            'augment': {'rescale': 1/255}}
    train = RandomBBoxGenerator(train_image_paths, state='train', seed=const.SEED, **params)
    val = RandomBBoxGenerator(val_image_paths, state='val', seed=const.SEED, **params)
    batch = train.__getitem__(0)
    create_samples(train)

