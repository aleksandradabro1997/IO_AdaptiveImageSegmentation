import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def check_if_file_exists(path: str) -> bool:
    """
    Check if particular file does exist.
    :param path: Path to file
    :return: True/False
    """
    if os.path.isfile(path):
        return True
    else:
        return False


def visualize_based_on_label(labels: np.array) -> None:
    """
    Plot labelled picture.
    :param labels:
    :return:
    """
    plt.imshow(labels)
    plt.title('Best result')