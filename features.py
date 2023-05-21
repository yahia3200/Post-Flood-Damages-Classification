import pandas as pd
import numpy as np
import cv2
from skimage import color
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def histogram_features(image, bins=15):
    """
    Compute the histogram of each color channel and concatenate them into a single feature vector.
    :param image: The input image.
    :param bins: The number of bins to use in the histogram.
    :return: The concatenated histogram feature vector.
    """
    red = np.histogram(image[:, :, 0], bins=bins)[0]
    green = np.histogram(image[:, :, 1], bins=bins)[0]
    blue = np.histogram(image[:, :, 2], bins=bins)[0]
    return np.concatenate((red, green, blue))


def glcm_features(image, features=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
                  distance=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Compute the gray-level co-occurrence matrix (GLCM) of the image and extract features from it.
    :param image: The input image.
    :param distance: The distance between the two pixels.
    :param angles: The angles between the two pixels.
    :return: The GLCM features.
    """

    image = color.rgb2gray(image)
    image = image * 255
    image = image.astype('uint8')
    glcm = graycomatrix(image, distances=distance, angles=angles)

    calculate_features = []

    # calculate features
    for feature in features:
        calculate_features.append(graycoprops(glcm, feature))
    # add features to a vector 
    result = {}

    for i in range(len(distance)):
        for j in range(len(angles)):
            for feature in range(len(features)):
                result[features[feature] + '_' + str(i) + '_' + str(j)] = calculate_features[feature][i, j]
    return result

def sift_features(image):
    """
    Compute the SIFT features of the image.
    :param image: The input image.
    :return: The SIFT features.
    """
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
    return des