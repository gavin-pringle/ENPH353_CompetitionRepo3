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


#DataCrop
path = "/home/fizzer/cnn_repo/ENPH353_CompetitionRepo3/"

_,_, files = next(os.walk(path + "output/")) # list of strings of file names in /output
file_count = len(files)

numberCharacters = 5

width = 600
height = 1800

darken = randint(60,150)/50

for i in range(file_count):
    picture = cv2.imread(path + "output/" + files[i])

    img_hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)

    img_hsv[:, :, 2] = img_hsv[:, :, 2] / darken

    picture = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    blur = randint(0,3)*2+11
    picture = cv2.GaussianBlur(picture, (blur,blur), 5)

    license = picture[1250:1550, 0:width]

    LicensePos = np.empty((4, 140, 100, 3))

    LicensePos[0] = license[90:230, 50:150]
    LicensePos[1] = license[90:230, 150:250]
    LicensePos[2] = license[90:230, 345:445]
    LicensePos[3] = license[90:230, 445:545]


    parkingSpace = picture[700:1100, int(width/2):width]

    platename = files[i]

    for j in range(numberCharacters-1):
        cv2.imwrite(os.path.join(path + "croppedPictures/", 
                                "plate_{}_{}.png".format(platename[j+3], i)),
                                LicensePos[j])
    
    cv2.imwrite(os.path.join(path + "croppedPictures/", 
                                "plate_{}_{}.png".format(platename[1], i)),
                                parkingSpace)