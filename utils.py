import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from typing import Dict
from matplotlib.backends.backend_pdf import PdfPages


def check_if_file_exists(path: str) -> bool:
    """Check if file exists

    :param path: path to file
    :return: True if file exists, otherwise False
    """
    if os.path.isfile(path):
        return True
    else:
        return False


def visualize_based_on_label(labels: np.array, original: np.array) -> None:
    """Plot segmented image

    :param original: Original image
    :param labels: Segmented image
    """
    plt.figure()
    plt.subplot(121)
    io.imshow(color.label2rgb(labels))
    plt.title('Best result')
    plt.subplot(122)
    plt.imshow(original)
    plt.title('Orginal image')
    plt.show()


def compare_with_ground_truth(segmented: np.array, gt: np.array) -> float:
    """Compare segmentation result with ground truth and return score.

    :param segmented: algorithm result
    :param gt: ground truth image
    :return: score
    """

    score = sum(segmented == gt)/(segmented.shape[0]*segmented.shape[1])

    return score


def print_results(scores: Dict) -> None:
    """Print scores for each image in dataset

    :param scores: dictionary with results
    :return: None
    """
    print(f'\nRUN RESULTS\n')
    print(f' image_name    score')
    for img_name, score in scores.items():
        print(f'{img_name}:    {score}')


def plot_fitness_function_per_image(qualities: Dict, img_name: str) -> plt.figure:
    """Plot fitness function value, over iterations for each image.

    :param img_name: name of the image
    :param qualities: fitness function values from algorithm
    :return: plot
    """
    fig = plt.figure()
    plt.plot(range(1, len(qualities)+1), [max(x['quality']) for x in qualities], 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness function value')
    plt.grid()
    plt.title(f'{img_name}')
    return fig


def plot_to_pdf(gt: Dict, algo_output: Dict, qualities: Dict) -> None:
    """Save plots to pdf.

    :param algo_output: segmented images
    :param gt: ground truth values
    :param qualities: values from algorithm
    :return: None
    """
    figs = []
    for img_name, value in qualities.items():
        figs.append(plot_fitness_function_per_image(qualities[img_name], img_name))
        figs.append(plot_gt_and_result(gt[img_name][0], algo_output[img_name][0]))

    report_name = f"ga_{datetime.datetime.now().strftime('%Y_%m_%dT%H_%M')}.pdf"
    pp = PdfPages(f'results/{report_name}')
    for fig in figs:
        pp.savefig(fig)
    pp.close()


def plot_gt_and_result(gt_image: np.array, algo_output: np.array) -> plt.figure:
    """Plot result and ground truth next to each other

    :param gt_image: ground truth array
    :param algo_output: algorithm result
    :return: figure
    """
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(color.label2rgb(gt_image))
    plt.title('Ground truth')
    plt.subplot(1, 2, 2)
    plt.imshow(color.label2rgb(algo_output))
    plt.title('Algorithm output')
    return fig