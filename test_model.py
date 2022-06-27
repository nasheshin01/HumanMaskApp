import keras
import cv2
import os


import tensorflow as tf
import numpy as np


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

def main():
    image = cv2.imread("data\images\HipHop_HipHop1_C0_00180.png")
    image_norm = image / 255.0

    model = Model()
    model.build(input_shape=(1, 194, 259, 3))
    model.load_weights("model.h5")

    mask = model.predict(np.array([image_norm]))
    cv2.imwrite("test_mask.jpg", (mask[0] * 256).astype(int))

main()