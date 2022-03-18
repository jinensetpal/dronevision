#!/usr/bin/env python3

from tensorflow.keras.layers import Conv2D, Flatten, Dense
from src.data.generator import RandomBBoxGenerator, get_image_paths
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
from tensorflow.keras import layers
import tensorflow as tf
from src import const
import numpy as np
import os

def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(Input(shape=input_shape, batch_size=const.BATCH_SIZE))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    # model.add(Dense(1, kernel_regularizer=l2(0.01), activation = "linear"))
    model.add(Dense(1, activation='softmax'))

    return model

if __name__ == '__main__':
    train_image_paths, val_image_paths = get_image_paths()
    params = {'base_dir': os.path.join(os.getcwd(), 'data'),
            'dim': const.TARGET_SIZE,
            'batch_size': const.BATCH_SIZE,
            'seed': const.SEED,
            'n_channels': 3,
            'shuffle': True,
            'augment': {'rescale': 1/255}}
    train = RandomBBoxGenerator(train_image_paths, state='train', **params)
    val = RandomBBoxGenerator(val_image_paths, state='val', **params)
    
    print(train.__getitem__(0)[0].shape)

    model = build_model(const.TARGET_SHAPE)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(train,
                        epochs=const.EPOCHS,
                        validation_data=val,
                        verbose=1)
    model.save(os.path.join('models', 'svm'))
