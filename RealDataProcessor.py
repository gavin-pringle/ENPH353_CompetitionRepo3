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
path = "/home/fizzer/cnn_repo/ENPH353_CompetitionRepo3/RealWorldData/"

_,_, files1 = next(os.walk(path + "license/")) # list of strings of file names in /output
file_count = len(files1)

_,_, files2 = next(os.walk(path + "parkingNumber/")) # list of strings of file names in /output

numberCharacters = 5


for i in range(file_count):
    
    _,_, files = next(os.walk(path + "ProcessedDataFolder/")) # list of strings of file names in /output
    count = len(files)

    license = cv2.imread(path + "license/" + files1[i])
    ParkingSpace = cv2.imread(path + "parkingNumber/" + files2[i])

    licenseName = files1[i]
    parkingName = files2[i]

    LicensePos = np.empty((5, 170, 100, 3))

    LicensePos[0] = license[80:250, 50:150]
    LicensePos[1] = license[80:250, 150:250]
    LicensePos[2] = license[80:250, 345:445]
    LicensePos[3] = license[80:250, 345:445]
    LicensePos[4] = license[80:250, 445:545]

    for j in range(numberCharacters):
        if(j!= 2):
            cv2.imwrite(os.path.join(path + "ProcessedDataFolder/", 
                                "plate_{}_{}.png".format(licenseName[j], count+j)),
                                LicensePos[j])
    
    cv2.imwrite(os.path.join(path + "ProcessedDataFolder/", 
                                "plate_{}_{}.png".format(parkingName[0], count)),
                                ParkingSpace)