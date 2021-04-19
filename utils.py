import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage import segmentation


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
    plt.imshow(orginal)
    plt.title('Orginal image')
    plt.show()
