from skimage import io
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import numpy as np
from typing import Dict, Tuple
import cv2


def run_slic_algorithm(image: np.array, nb_of_clusters: int, image_size: Tuple, nb_of_iterations: int) -> np.array:
    """
    Runs SLIC algorithm from skimage
    :param image - input image
    :param nb_of_clusters - number of output image segments according to ground truth
    :return: segmentation result
    """
    image = cv2.resize(image, image_size)
    img_slic = seg.slic(image, n_segments=nb_of_clusters, compactness=10, max_iter=nb_of_iterations, convert2lab=True, sigma=1)
    return img_slic


def compare_results_slic(slic_output: Dict, ground_truth: Dict) -> Dict:
    """
    Compare SLIC output with ground truth and return score.
    :param slic_img: output of SLIC algorithm
    :param ground_truth: ground truth values
    :return: score
    """
    scores = {}
    for img_name, gt in ground_truth.items():
        try:
            slic_img = slic_output[img_name]
            gt = ground_truth[img_name][0]
            score = np.sum(gt == slic_img)
            scores.update({img_name: score/(slic_img.shape[0] * slic_img.shape[1])})
        except:
            continue
    return scores


def run_k_means(image: np.array, nb_of_clusters: int, image_size: Tuple, nb_of_iterations: int) -> np.array:
    """Segment image using k-means algorithm

    :param image: input image
    :param nb_of_clusters: number of segments
    :param image_size: size of image
    :return: segmented image
    """
    image = cv2.resize(image, image_size)
    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = nb_of_clusters
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, nb_of_iterations, cv2.KMEANS_RANDOM_CENTERS)
    return labels.reshape(image_size)
