import cv2
import numpy as np
from scipy.stats import skew, kurtosis

def mean_std(image):
    means, stddevs  = cv2.meanStdDev(image)
    return means.reshape(-1), stddevs.reshape(-1)

def get_skewness_kurt(image):
    skewness = []
    kurt = []
    try:
        shape = image.shape[2]
    except:
        shape = 1
    for channel in range(0,shape):        
        skewness.append(skew(image.reshape(-1,shape)[:,channel]))
        kurt.append(kurtosis(image.reshape(-1,shape)[:,channel]))
    return skewness, kurt

def extract_color_features(image):
    color_features = []
    name_features = []

    ## Original image in RGB
    means, stdvs = mean_std(image)
    skewness, kurts = get_skewness_kurt(image)

    color_features.extend(means)
    color_features.extend(stdvs)
    color_features.extend(skewness)
    color_features.extend(kurts)
    
    name_features.extend(['rgb_mean_0','rgb_mean_1','rgb_mean_2', 
                          'rgb_std0','rgb_std1','rgb_std2', 
                          'rgb_skew0','rgb_skew1','rgb_skew2', 
                          'rgb_kurt0', 'rgb_kurt0', 'rgb_kurt0'])

    ## Image in HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    means, stdvs = mean_std(image_hsv)
    skewness, kurts = get_skewness_kurt(image_hsv)

    color_features.extend(means)
    color_features.extend(stdvs)
    color_features.extend(skewness)
    color_features.extend(kurts)

    name_features.extend(['hsv_mean_0','hsv_mean_1','hsv_mean_2', 
                          'hsv_std0','hsv_std1','hsv_std2', 
                          'hsv_skew0','hsv_skew1','hsv_skew2', 
                          'hsv_kurt0', 'hsv_kurt1', 'hsv_kurt2'])

    ## Image in L*a*b

    image_LAB = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2LAB)

    means, stdvs = mean_std(image_LAB)
    skewness, kurts = get_skewness_kurt(image_LAB)

    color_features.extend(means)
    color_features.extend(stdvs)
    color_features.extend(skewness)
    color_features.extend(kurts)

    name_features.extend(['lab_mean_0','lab_mean_1','lab_mean_2', 
                          'lab_std0','lab_std1','lab_std2', 
                          'lab_skew0','lab_skew1','lab_skew2', 
                          'lab_kurt0', 'lab_kurt1', 'lab_kurt2'])    

    ## Image in YCbCr

    image_YCC = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

    means, stdvs = mean_std(image_YCC)
    skewness, kurts = get_skewness_kurt(image_YCC)

    color_features.extend(means)
    color_features.extend(stdvs)
    color_features.extend(skewness)
    color_features.extend(kurts)

    name_features.extend(['ycc_mean_0','ycc_mean_1','ycc_mean_2', 
                          'ycc_std0','ycc_std1','ycc_std2', 
                          'ycc_skew0','ycc_skew1','ycc_skew2', 
                          'ycc_kurt0', 'ycc_kurt1', 'ycc_kurt2'])    
    ## Image in Grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    means, stdvs = mean_std(gray_img)
    skewness, kurts = get_skewness_kurt(gray_img)

    color_features.extend(means)
    color_features.extend(stdvs)
    color_features.extend(skewness)
    color_features.extend(kurts)

    name_features.extend(['gray_mean','gray_std','gray_skew','gray_kurt'])

    dict_features = {}
    
    for i in range(len(name_features)):
        dict_features[name_features[i]] = color_features[i]
    
    return dict_features    
