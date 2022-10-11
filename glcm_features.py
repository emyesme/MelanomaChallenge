import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
import itertools
from copy import deepcopy

def get_glcm(image, angles, distances, colorspace):

    # combination of distances and angles as couples of values
    distancesAngles = list(itertools.product(distances, angles))
    dictFeatures = {}
    image_org = deepcopy(image)
    
    if colorspace == 'rgb':
        channels = 3
    elif colorspace == 'hsv':
        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2HSV)
        channels = 3
    elif colorspace == 'lab':
        image = cv2.cvtColor(np.uint8(image_org), cv2.COLOR_RGB2LAB)
        channels = 3
    elif colorspace == 'ycc':
        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2YCR_CB)
        channels = 3
    elif colorspace == 'gray':
        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
        channels = 1
        
    for channel in range(channels):
        for distanceAngle in distancesAngles:
            distance = distanceAngle[0]
            angle = distanceAngle[1]
            # get the degree to use it as name for the column
            name = str(angle*(180.0/np.pi))
            try:
                glcm = greycomatrix(image[:, :, channel], [ distance ], [ angle ])
            except:
                glcm = greycomatrix(image, [ distance ], [ angle ])

            # properties: {‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’}
            dictFeatures['contrast'+ str(distance) + name + colorspace + str(channel)] = greycoprops(glcm, 'contrast')[0][0]
            dictFeatures['dissimilarity' + str(distance) + name + colorspace + str(channel)] = greycoprops(glcm, 'dissimilarity')[0][0]
            dictFeatures['homogeneity' + str(distance) + name + colorspace + str(channel)] = greycoprops(glcm, 'homogeneity')[0][0]
            dictFeatures['energy' + str(distance) + name + colorspace + str(channel)] = greycoprops(glcm, 'energy')[0][0]
            dictFeatures['correlation' + str(distance) + name + colorspace + str(channel)] = greycoprops(glcm, 'correlation')[0][0]
            dictFeatures['ASM' + str(distance) + name + colorspace + str(channel)] = greycoprops(glcm, 'ASM')[0][0]
    
    return dictFeatures