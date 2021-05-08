import numpy as np
from typing import Dict, Tuple
from skimage import metrics
from internals import assign_labels_based_on_chromosome


def compare_results(ground_truth: Dict, algo_output: Dict, image_size: Tuple, images: Dict):
    """Compare both outputs and return score

    :param images: orginal images
    :param image_size: size of input to algorithm specified by user in args
    :param ground_truth: Dict with names and gt
    :param algo_output: dict with output
    :return: Dictionary name : score
    """
    scores = {}
    for img_name, gt in ground_truth.items():
        try:
            algo_img = algo_output[img_name][0]
            algo_chromosome = algo_output[img_name][1]
            gt = ground_truth[img_name][0]
            best_lab, best_dist = assign_labels_based_on_chromosome(algo_chromosome, image_size, images[img_name])
            score = 0
            for i in range(ground_truth[img_name][1]):
                label_indexes = np.where(algo_img == i)
                centre_index = np.argmin(best_dist[algo_img == 0]) #np.where(best_dist[algo_img == 0] == 0)[0]
                label_gt = gt[label_indexes[0][centre_index], label_indexes[1][centre_index]]
                tmp_gt = np.ones(image_size) * 100
                tmp_gt[np.where(gt == label_gt)] = label_gt
                score += np.sum(tmp_gt == algo_img)

            scores.update({img_name: score/(image_size[0] * image_size[1] * 3)})
        except:
            continue
    return scores


def calculate_adapted_rand_error(ground_truth: Dict, algo_output: Dict) -> Dict:
    """
    Compute Adapted Rand error as defined by the SNEMI3D contest
    :param ground_truth: Dict with names and gt
    :param algo_output: Dict with Adaptive Segmentation results
    :return: are scores
    """
    are_scores = {}
    for img_name, gt in ground_truth.items():
        try:
            scores = metrics.adapted_rand_error(ground_truth[img_name][0], algo_output[img_name][0])
            are_scores.update({img_name: scores})
        except:
            continue
    return are_scores


def calculate_adapted_rand_error_slic(ground_truth: Dict, algo_output: Dict) -> Dict:
    """
    Compute Adapted Rand error as defined by the SNEMI3D contest
    :param ground_truth: Dict with names and gt
    :param algo_output: Dict with Adaptive Segmentation results
    :return: are scores
    """
    are_scores = {}
    for img_name, gt in ground_truth.items():
        try:
            scores = metrics.adapted_rand_error(ground_truth[img_name][0], algo_output[img_name])
            are_scores.update({img_name: scores})
        except:
            continue
    return are_scores
