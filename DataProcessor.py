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

def gaussian_noisied(input):
    gauss = np.random.normal(-1,1,input.size)
    gauss = gauss.reshape(input.shape[0],input.shape[1]).astype('uint8')
    return cv2.add(input,gauss)


#DataCrop
path = "/home/fizzer/cnn_repo/ENPH353_CompetitionRepo3/"

_,_, files = next(os.walk(path + "output/")) # list of strings of file names in /output
file_count = len(files)

numberCharacters = 5

width = 600
height = 1800


for i in range(file_count):
    darken = randint(60,150)/50
    picture = cv2.imread(path + "output/" + files[i])

    img_hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)

    img_hsv[:, :, 2] = img_hsv[:, :, 2] / darken

    picture = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    blur = randint(0,3)*2+11
    picture = cv2.blur(picture, (blur,blur), 5)

    license = picture[1250:1550, 0:width]
    parkingSpace = picture[700:1100, int(width/2):width]


    hsvLicense = cv2.cvtColor(license, cv2.COLOR_BGR2HSV)

    uh = 122
    us = 255
    uv = 255
    lh = 118
    ls = 80
    lv = 0
    lower_hsv = np.array([lh,ls,lv])
    upper_hsv = np.array([uh,us,uv])

    WhiteLicense = cv2.inRange(hsvLicense, lower_hsv, upper_hsv)

    hsvParking = cv2.cvtColor(parkingSpace, cv2.COLOR_BGR2HSV)

    uh = 5
    us = 5
    uv = 50
    lh = 0
    ls = 0
    lv = 0
    lower_hsv = np.array([lh,ls,lv])
    upper_hsv = np.array([uh,us,uv])

    hsvParking = cv2.inRange(hsvParking, lower_hsv, upper_hsv)


    WhiteLicense = cv2.blur(WhiteLicense, (blur,blur), 5)
    hsvParking = cv2.blur(hsvParking, (blur,blur), 5)

    LicensePos = np.empty((4, 170, 100))

    LicensePos[0] = WhiteLicense[80:250, 50:150]
    LicensePos[1] = WhiteLicense[80:250, 150:250]
    LicensePos[2] = WhiteLicense[80:250, 345:445]
    LicensePos[3] = WhiteLicense[80:250, 445:545]

    hsvParking = cv2.resize(hsvParking,(100,170))


    platename = files[i]

    for j in range(numberCharacters-1):
        cv2.imwrite(os.path.join(path + "croppedPictures/", 
                                "plate_{}_{}.png".format(platename[j+3], i)),
                                LicensePos[j])
    
    cv2.imwrite(os.path.join(path + "croppedPictures/", 
                                "plate_{}_{}.png".format(platename[1], i)),
                                hsvParking)