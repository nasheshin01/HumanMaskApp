from math import fabs
import keras
import cv2
import os

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


MASKS_PATH = "data\\masks"
IMAGES_PATH = "data\\images"


def join_paths(origin_path, filenames):
    return [os.path.join(origin_path, filename) for filename in filenames]


def read_images(path: str, is_test: bool):
    filenames = os.listdir(path)
    filepaths = join_paths(path, filenames)

    images = []
    for filepath in filepaths:
        image = cv2.imread(filepath)
        if is_test:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_norm = image / 255.0
        images.append(image_norm)

    return np.array(images)


class Model(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv2 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv3 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv4 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv5 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.conv6 = keras.layers.Conv2D(1, (3, 3), padding='same', activation=None)
        self.pool = keras.layers.MaxPool2D((2, 2))
        
    def call(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = tf.image.resize(out, (x.shape[1], x.shape[2]), tf.image.ResizeMethod.BILINEAR)
        out = tf.nn.sigmoid(out)
        return out
    
model = Model()


def main():
    x, y = read_images(IMAGES_PATH, False), read_images(MASKS_PATH, True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    model = Model()

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    hist = model.fit(x_train, y_train, epochs=10, batch_size=1, validation_data=(x_test, y_test))
    model.save_weights("model.h5")



if __name__ == '__main__':
    main()
