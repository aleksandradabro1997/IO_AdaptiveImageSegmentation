import cv2
import logging
import argparse
import numpy as np

from scipy.spatial import distance
from collections import defaultdict
from typing import List, Tuple
from enums import SelectionMethod, MutationType, CodingMethod
from utils import check_if_file_exists, visualize_based_on_label

parser = argparse.ArgumentParser(description='Adaptive Image Segmentation Algorithm')
parser.add_argument('--nb-of-iterations', default=10, type=int,
                    help='Number of algorithms iterations')
parser.add_argument('--nb-of-clusters', default=10, type=int,
                    help='Number of clusters for the image')
parser.add_argument('--coding-method', default=0, type=int,
                    help='One of 3 coding types (classic, permutational, woody)')
parser.add_argument('--selection-method', default=0, type=int,
                    help='Type of selection method (roulette, rank, tournament)')
parser.add_argument('--mutation-type', default=0, type=int,
                    help='Mutation type')
parser.add_argument('--crossover-rate', default=0.1, type=float,
                    help='Value of crossover rate')
parser.add_argument('--image-path', default='images/Lenna.png', type=str,
                    help='Path to image to be segmented')

#args = parser.parse_args()

# Set random seed
np.random.seed(seed=1)

# Configure logger
logging.basicConfig(level=logging.DEBUG)

# Constants
POPULATION_SIZE = 100


def generate_initial_population(population_size: int, number_of_clusters: int,
                                image_size: Tuple, image: np.array) -> List:
    """
    Generates population at the start
    :param population_size: number of chromosomes
    :param number_of_clusters: number of areas that picture must be divided into
    :return: list of chromosomes in population
    """
    population = []
    for i in range(population_size + 1):
        chromosome = defaultdict()
        for j in range(0, number_of_clusters + 1):
            # Generate cluster centre
            x = np.random.randint(0, image_size[0])
            y = np.random.randint(0, image_size[1])
            cluster = image[x, y, :]
            chromosome[j] = cluster
        population.append(chromosome)
    return population


def assign_labels_based_on_chromosome(chromosome: List, image_size: Tuple, image: np.array) -> Tuple:
    """
    Assign label to each pixel in the picture based on cluster centres and euclidean distance
    :param chromosome: one possible solution
    :param image_size: dimensions of an image
    :return:
    """
    labels = np.ones(image_size) * (-1)
    centre_distances = np.ones(image_size) * (-1)
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            min_distance = 1e100
            min_cluster_label = -1
            for k in range(len(chromosome)):  # k-cluster
                # Calculate distance
                tmp_dist = distance.euclidean(chromosome[k], image[i, j, :])
                if tmp_dist < min_distance:
                    min_distance = tmp_dist
                    min_cluster_label = k
            # Set label
            labels[i, j] = min_cluster_label
            centre_distances[i, j] = min_distance

    return np.array(labels, dtype=int), centre_distances


def calculate_fitness_function(chromosome: List, image_size: Tuple, image: np.array) -> float:
    """
    Calculate quality.
    Sum of min distances in each cluster in each chromosome
    :param chromosome: one particular solution
    :param image_size: image dimension
    :param image: image to be segmented
    :return: value of fitness function
    """
    quality = 0
    labels, centre_distances = assign_labels_based_on_chromosome(chromosome, image_size, image)
    # There where centre_distances == 0 set high value
    centre_distances = np.where(centre_distances == 0, 1e100, centre_distances)
    for k in range(len(chromosome)):
        min_distance = np.amin(centre_distances[labels == k])
        quality += min_distance
    return quality


def generate_new_population(reproductive_group: List, method: CodingMethod) -> List:
    """
    Generate new population, based on chromosomes selected from current population
    :param reproductive_group: group with best, according to fitness function, solutions
    :param method: the way of coding
    :return: new population
    """
    if method == CodingMethod.CLASSIC:
        pass
    elif method == CodingMethod.PERMUTATIONAL:
        pass
    elif method == CodingMethod.WOODY:
        pass
    else:
        raise ValueError(f'Invalid coding method passed!')


def select_reproductive_group(current_population: List, qualities: List, method: SelectionMethod) -> List:
    """
    Select reproductive group according to qualities and selected method
    :param current_population: current group of solutions
    :param qualities: values of fitness function
    :param method: Method of selection
    :return: best possible solutions in population
    """
    if method == SelectionMethod.RANK:
        pass
    elif method == SelectionMethod.ROULETTE:
        pass
    elif method == SelectionMethod.TOURNAMENT:
        pass
    else:
        raise ValueError(f'Invalid selection method passed!')


def perform_mutation(new_population: List, method: MutationType) -> List:
    """
    Mutate some of the chromosomes.
    :param new_population: generated new solutions
    :param method: type of the mutation
    :return: mutated solution
    """
    if method == MutationType.INVERSION:
        pass
    elif method == MutationType.REMOVAL:
        pass
    elif method == MutationType.SUBSTITUTION:
        pass
    else:
        raise ValueError(f'Invalid mutation type passed!')


def check_end_criterion() -> bool:
    """
    Check if algorithm can be ended.
    :return:
    """
    raise NotImplementedError


def get_final_result() -> np.array:
    """
    Get best image.
    :return: - segmented image.
    """
    raise NotImplementedError


def ga_segmentation(image: str, nb_of_iterations: int, population_size: int, nb_of_clusters: int,
                    coding_type: CodingMethod, selection_method: SelectionMethod,
                    mutation_type: MutationType, crossover_rate: float) -> np.array:
    """
    Main algorithm function
    :param image: path to image to be segmented
    :param nb_of_iterations: max number of algorithm iterations
    :param nb_of_clusters:
    :param coding_type: coding method to be used
    :param selection_method: selection method to be used
    :param mutation_type: mutation type to be used
    :param crossover_rate:
    :return: None
    """
    # Read image if exists
    if not check_if_file_exists(image):
        raise FileNotFoundError(f'File {image} does not exist.')

    image_rgb = cv2.imread(image)
    image_size = image_rgb.shape[0:2]

    # Generate initial population
    population_init = generate_initial_population(population_size=population_size,
                                                  number_of_clusters=nb_of_clusters,
                                                  image_size=image_size,
                                                  image=image_rgb)
    population = population_init
    for iteration in range(nb_of_iterations):
        # Calculate quality for each element in population
        qualities = []
        for chromosome in population:
            quality = calculate_fitness_function(population, image_size, image_rgb)
            qualities.append(quality)
        reproductive_group = select_reproductive_group(population, qualities, selection_method)
        new_population = generate_new_population(reproductive_group, coding_type)
        new_population_mutated = perform_mutation(new_population, mutation_type)
        criterion_ok = check_end_criterion()
        if criterion_ok:
            segmented_image = get_final_result()
            break

    return segmented_image


if __name__ == '__main__':
    best = ga_segmentation(image=args.image_path,
                           nb_of_iterations=args.nb_of_iterations,
                           population_size=POPULATION_SIZE,
                           nb_of_clusters=args.nb_of_clusters,
                           coding_type=args.coding_method,
                           selection_method=args.selection_method,
                           mutation_type=args.mutation_type,
                           crossover_rate=args.crossover_rate)

    visualize_based_on_label(best)
