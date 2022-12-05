#!/usr/bin/env python3
from __future__ import print_function
import string
import random
from random import randint
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw
import math
import re
from collections import Counter
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
import roslib
import sys
from geometry_msgs.msg import Twist
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ConvNet:
    def __init__(self):
        cv2.waitKey(1)


    def characterToOneHot(chara):
        characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        for i in range(len(characters)):
            if (chara == characters[i]):
                arr = np.zeros(len(characters))
                arr[i] = 1
                return arr
            i += 1

    def characterFromOneHot(chara):
        characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return characters[chara]

    def dataLabel(self):
        path = "/home/fizzer/cnn_repo/ENPH353_CompetitionRepo3/"

        _,_, files = next(os.walk(path + "croppedPictures/")) # list of strings of file names in /pictures
        file_count = len(files)

        totalChar = 36

        X = []
        Y = []

        for i in range(file_count):
            plateChar = files[i]
            Char = plateChar[6]
            X.append(cv2.imread(path + "croppedPictures/" + files[i]))
            Y.append(self.characterToOneHot(Char))

        X = np.asarray(X)
        Y = np.asarray(Y)

        X = X/255

        return X,Y 

    def reset_weights(model):
        for ix, layer in enumerate(model.layers):
            if (hasattr(model.layers[ix], 'kernel_initializer') and 
                hasattr(model.layers[ix], 'bias_initializer')):
                weight_initializer = model.layers[ix].kernel_initializer
                bias_initializer = model.layers[ix].bias_initializer

                old_weights, old_biases = model.layers[ix].get_weights()

                model.layers[ix].set_weights([
                    weight_initializer(shape=old_weights.shape),
                    bias_initializer(shape=len(old_biases))])

    def initializeModel():
        conv_model = models.Sequential()
        conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                    input_shape=(150, 150, 3)))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Flatten())
        conv_model.add(layers.Dropout(0.5))
        conv_model.add(layers.Dense(512, activation='relu'))
        conv_model.add(layers.Dense(36, activation='softmax'))

        return conv_model

    def compileModel(conv_model):
        LEARNING_RATE = 1e-4
        conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])

    def runModel(conv_model, X, Y):
        VALIDATION_SPLIT = 0.2
        history_conv = conv_model.fit(X, Y, 
                                    validation_split=VALIDATION_SPLIT, 
                                    epochs=80, 
                                    batch_size=16)


## Main function that keeps the simulation running until it is stopped by a user
def main(args):
  # name of node is line_follow
  rospy.init_node('line_follow', anonymous=True)
  ic = ConvNet()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)