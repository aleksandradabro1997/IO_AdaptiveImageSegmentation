from skimage import io
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import numpy as np
from typing import Dict, Tuple
import cv2


def run_slic_algorithm(image: np.array, nb_of_clusters: int, image_size: Tuple) -> np.array:
    """
    Runs SLIC algorithm from skimage
    :param image - input image
    :param nb_of_clusters - number of output image segments according to ground truth
    :return: segmentation result
    """
    image = cv2.resize(image, image_size)
    img_slic = seg.slic(image, n_segments=nb_of_clusters, compactness=10, max_iter=10, convert2lab=True, sigma=1)
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