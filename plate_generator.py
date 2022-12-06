#!/usr/bin/env python3

import cv2
import csv
import numpy as np
import os
import random
import string
from random import randint
from PIL import Image, ImageFont, ImageDraw

path = os.path.dirname(os.path.realpath(__file__)) + "/"
output_path = "output/"

with open(path + "plates.csv", 'w') as plates_file:
    csvwriter = csv.writer(plates_file)

    # Number of sets of 10 plates that are needed
    for multiplier in range(0, 50):
        # Want P1 through P9
        for i in range(0, 9):

            # Pick two random letters
            plate_alpha = ""
            for _ in range(0, 2):
                plate_alpha += (random.choice(string.ascii_uppercase))
            num = randint(0, 99)

            # Pick two random numbers
            plate_num = "{:02d}".format(num)

            # Create parking spot label
            s = "P" + str(i+1)

            # Save plate to file
            csvwriter.writerow([s+"_"+plate_alpha+plate_num])

            # Write plate to image
            blank_plate = cv2.imread(path+'blank_plate.png')

            # To use monospaced font for the license plate we need to use the PIL
            # package.
            # Convert into a PIL image (this is so we can use the monospaced fonts)
            blank_plate_pil = Image.fromarray(blank_plate)
            # Get a drawing context
            draw = ImageDraw.Draw(blank_plate_pil)
            monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 200)
            draw.text((48, 105),plate_alpha + " " + plate_num, (255,0,0), font=monospace)
            # Convert back to OpenCV image and save
            blank_plate = np.array(blank_plate_pil)

            parking_spot = 255 * np.ones(shape=[600, 600, 3], dtype=np.uint8)
            cv2.putText(parking_spot, s, (30, 450), cv2.FONT_HERSHEY_PLAIN, 28,
                        (0, 0, 0), 30, cv2.LINE_AA)
            spot_w_plate = np.concatenate((parking_spot, blank_plate), axis=0)

            # Merge labelled images and save
            labelled = np.concatenate((255 * np.ones(shape=[600, 600, 3],
                                        dtype=np.uint8), spot_w_plate), axis=0)
            cv2.imwrite(os.path.join(path+output_path, s+"_"+plate_alpha+plate_num+".png"), labelled)
