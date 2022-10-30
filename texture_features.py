import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp(img, radius, points): #radius 1 and points 8
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_img, points, radius, method='default')

    n_bins = int(points*(points- 1)+3)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    
    featuresDict = {}
    for i in range(len(hist)):
        featuresDict['lbp_'+str(i)] = hist[i] 
    
    return featuresDict


def extract_orb(img, featuresNumber): # should be 64
    
# ORB: an efficient alternative to SIFT or SURF
# @inproceedings{rublee2011orb,
#   title={ORB: An efficient alternative to SIFT or SURF},
#   author={Rublee, Ethan and Rabaud, Vincent and Konolige, Kurt and Bradski, Gary},
#   booktitle={2011 International conference on computer vision},
#   pages={2564--2571},
#   year={2011},
#   organization={Ieee}
# }
    
    orb = cv2.ORB_create(nfeatures = featuresNumber, fastThreshold=0, edgeThreshold=0)

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    
    print(len(des))
    print(des)
    
    features = []
    for vector in des:
        features.extend(vector)
            
    dict_features = {}
        
    for i in range(len(features)):
        dict_features['orb_'+str(i)] = features[i]
    
    
    return dict_features