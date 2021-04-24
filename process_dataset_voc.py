import os
import numpy as np
from typing import Dict, Tuple
import cv2


# BGR to labels mapping
LABEL_1 = (192, 224, 224)
LABEL_2 = (0, 192, 128)
LABEL_3 = (128, 64, 0)
LABEL_4 = (0, 0, 128)
LABEL_5 = (0, 0, 192)
LABEL_6 = (128, 0, 64)
LABEL_7 = (0, 128, 128)
LABEL_8 = (128, 128, 192)
LABEL_9 = (0, 64, 128)
LABEL_10 = (128, 0, 128)
LABEL_11 = (0, 128, 192)


def produce_ground_truth(dataset_path: str, size: Tuple) -> Dict:
    """Produce ground truth based on images in SegmentationObject

    :param dataset_path: path to VOC dataset
    :return: dictionary with name and ground truth
    """

    gt = {}
    images = os.listdir(dataset_path)
    for img in images:
        img_path = os.path.join(dataset_path, img)

        if os.path.isfile(img_path):
            img_gt = cv2.imread(img_path)
            img_gt = cv2.resize(img_gt, size)
            label_seg = np.zeros((img_gt.shape[:2]), dtype=np.int)
            #label_seg[(img_gt == LABEL_1).all(axis=2)] = 1
            label_seg[(img_gt == LABEL_2).all(axis=2)] = 2
            label_seg[(img_gt == LABEL_3).all(axis=2)] = 3
            label_seg[(img_gt == LABEL_4).all(axis=2)] = 4
            label_seg[(img_gt == LABEL_5).all(axis=2)] = 5
            label_seg[(img_gt == LABEL_6).all(axis=2)] = 6
            label_seg[(img_gt == LABEL_7).all(axis=2)] = 7
            label_seg[(img_gt == LABEL_8).all(axis=2)] = 8
            label_seg[(img_gt == LABEL_9).all(axis=2)] = 9
            label_seg[(img_gt == LABEL_10).all(axis=2)] = 10
            label_seg[(img_gt == LABEL_11).all(axis=2)] = 11

            nb_of_clusters = len(np.unique(label_seg))
            gt.update({img.split('.')[0]: (label_seg, nb_of_clusters)})

    return gt


def get_number_of_elements_from_xml(xml_path: str) -> int:
    """Get number of elements in image based on *.xml file.

    :param xml_path: path to *.xml file with description
    :return: number of clusters on image
    """
    with open(xml_path, 'r') as xml:
        content = xml.readlines()
    nb_of_clusters = 0
    for line in content:
        if '<name>' in line:
            nb_of_clusters += 1

    return nb_of_clusters


def read_all_images(path: str) -> Dict:
    """ Read all files from directory in.

    :param path: path to directory with RGB images
    :return: dictionary name: image
    """
    images = {}
    images_paths = os.listdir(path)
    for img_path in images_paths:
        img = cv2.imread(os.path.join(path, img_path))
        images.update({img_path.split('.')[0]: img})

    return images


def get_images_and_gt_from_dataset(path: str, size: Tuple) -> Tuple:
    """Process dataset

    :param path: directory with dataset (with directories in and gt)
    :return: dictionary with gt and images
    """
    gt_path = os.path.join(path, 'gt')
    img_path = os.path.join(path, 'in')

    gt = produce_ground_truth(gt_path, size)
    images = read_all_images(img_path)

    return images, gt
