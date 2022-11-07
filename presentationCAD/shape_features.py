import mahotas


def extract_zernike_moments(img, radius):
# Image should be a grayscale image
# radius is the area considered around the center of mass of the image to extract the features
# It gives back a vector of 25 elements
# zernike_moments function receives (img, radius, degree). Degree is by default 8, that is what we will use based on Bansal et al 2022.
    
    # computing zernike moments
    value = mahotas.features.zernike_moments(img, radius)
    
    featuresDict = {}
    
    for i in range(len(value)):
        featuresDict['zm_'+str(i)] = value[i]
    
    return featuresDict