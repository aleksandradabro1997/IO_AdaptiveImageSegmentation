import cv2
import logging
import argparse
import numpy as np

from scipy.spatial import distance
from collections import defaultdict
from typing import List, Tuple
from enums import SelectionMethod, MutationType, CodingMethod
from utils import check_if_file_exists, visualize_based_on_label
from selection import select_by_rank_method, select_by_roulette_method, select_by_tournament_method
from coding import generate_new_population_classic, generate_new_population_permutational, generate_new_population_woody
from mutation import perform_mutation_inversion, perform_mutation_removal, perform_mutation_substitution

parser = argparse.ArgumentParser(description='Adaptive Image Segmentation Algorithm')
parser.add_argument('--nb-of-iterations', default=4, type=int,
                    help='Number of algorithms iterations')
parser.add_argument('--nb-of-clusters', default=10, type=int,
                    help='Number of clusters for the image')
parser.add_argument('--coding-method', default=CodingMethod.PUBLICATION, type=int,
                    help='One of 3 coding types (classic, permutational, woody)')
parser.add_argument('--selection-method', default=SelectionMethod.PUBLICATION, type=int,
                    help='Type of selection method (roulette, rank, tournament)')
parser.add_argument('--mutation-type', default=MutationType.PUBLICATION, type=int,
                    help='Mutation type')
parser.add_argument('--crossover-rate', default=0.1, type=float,
                    help='Value of crossover rate')
parser.add_argument('--image-path', default='images/Lenna.png', type=str,
                    help='Path to image to be segmented')
parser.add_argument('--image-size', default=(200, 200), type=tuple,
                    help='Size of the image to be segmented')

# Set random seed
np.random.seed(seed=1)

# Configure logger
logging.basicConfig(level=logging.INFO)


def generate_initial_population(population_size: int, number_of_clusters: int,
                                image_size: Tuple, image: np.array) -> List:
    """Generate possible solutions at the start of the algorithm

    :param population_size: number of chromosomes
    :param number_of_clusters: number of areas that picture must be divided into
    :param image_size: size of the input image
    :param image: algorithm's input image
    :return: list of possible solutions
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


def assign_labels_based_on_chromosome(chromosome: defaultdict, image_size: Tuple, image: np.array) -> Tuple:
    """Assign label to each pixel in the picture based on cluster centres and euclidean distance

    :param image: image to be segmented
    :param chromosome: one possible solution
    :param image_size: dimensions of an image
    :return: segmented image, distances from centre
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


def calculate_fitness_function(chromosome: defaultdict, image_size: Tuple, image: np.array) -> float:
    """Calculate quality. Sum of min distances in each cluster in each chromosome.

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
        # Replace chromosome centre with mean cluster values
        indices = np.where(labels == k)
        new_center = (int(sum(indices[0])/len(indices[0])), int(sum(indices[1])/len(indices[1])))
        chromosome[k] = image[new_center[0], new_center[1], :]
    return quality


def generate_new_population(reproductive_group: List, qualities: List, method: CodingMethod, population_size: int,
                            number_of_clusters: int, image_size: Tuple, image: np.array) -> List:
    """Generate new population, based on chromosomes selected from current population,
    according to publication best chromosome survives others are random

    :param population_size: number of solutions in population
    :param image: input image
    :param number_of_clusters: number of parts to divide picture
    :param image_size: size of the input  image
    :param qualities: values of the fitness function for each solution
    :param reproductive_group: group with best, according to fitness function, solutions
    :param method: the way of coding
    :return: new population
    """
    new_population = []
    if method == CodingMethod.CLASSIC:
        new_population = generate_new_population_classic(reproductive_group=reproductive_group)
    elif method == CodingMethod.PERMUTATIONAL:
        new_population = generate_new_population_permutational(reproductive_group=reproductive_group)
    elif method == CodingMethod.WOODY:
        new_population = generate_new_population_woody(reproductive_group=reproductive_group)
    elif method == CodingMethod.PUBLICATION:
        new_population = generate_initial_population(image_size=image_size,
                                                     image=image,
                                                     population_size=population_size,
                                                     number_of_clusters=number_of_clusters)
        new_population[0] = select_best_chromosome(new_population, qualities)  # according to publication best chromosome is kept
    else:
        raise ValueError(f'Invalid coding method passed!')
    return new_population


def select_reproductive_group(current_population: List, qualities: List, method: SelectionMethod) -> List:
    """Select reproductive group according to qualities and selected method

    :param current_population: current group of solutions
    :param qualities: values of fitness function
    :param method: Method of selection
    :return: best possible solutions in population
    """
    if method == SelectionMethod.RANK:
        reproductive_group = select_by_rank_method(population=current_population,
                                                   qualities=qualities)
    elif method == SelectionMethod.ROULETTE:
        reproductive_group = select_by_roulette_method(population=current_population,
                                                       qualities=qualities)
    elif method == SelectionMethod.TOURNAMENT:
        reproductive_group = select_by_tournament_method(population=current_population,
                                                         qualities=qualities)
    elif method == SelectionMethod.PUBLICATION:
        reproductive_group = current_population
    else:
        raise ValueError(f'Invalid selection method passed!')
    return reproductive_group


def perform_mutation(new_population: List, method: MutationType) -> List:
    """Mutate some of the chromosomes.

    :param new_population: generated new solutions
    :param method: type of the mutation
    :return: mutated solution
    """
    if method == MutationType.INVERSION:
        new_population = perform_mutation_inversion(population=new_population)
    elif method == MutationType.REMOVAL:
        new_population = perform_mutation_removal(population=new_population)
    elif method == MutationType.SUBSTITUTION:
        new_population = perform_mutation_substitution(population=new_population)
    elif method == MutationType.PUBLICATION:
        new_population = new_population
    else:
        raise ValueError(f'Invalid mutation type passed!')
    return new_population


def check_end_criterion() -> bool:
    """Check if algorithm can be ended.

    :return: True if criterion passed, otherwise False
    """
    # According to publication the number of iterations is predefined
    pass


def select_best_chromosome(population: List, qualities: List) -> defaultdict:
    """Select best solution based on fitness function.

    :param population: current group of solutions
    :param qualities: list of qualities for each solution
    :return: best chromosome according to quality
    """
    best_chromosome = population[qualities.index(max(qualities))]
    return best_chromosome


def get_final_result(population: List, image_size: Tuple, image: np.array) -> np.array:
    """Get best image.

    :param image:
    :param image_size:
    :param population: object
    :return: segmented image.
    """
    qualities = []
    for chromosome in population:
        quality = calculate_fitness_function(chromosome, image_size, image)
        qualities.append(quality)

    best_chromosome = select_best_chromosome(population, qualities)
    best_segmentation = assign_labels_based_on_chromosome(best_chromosome, image_size, image)
    return best_segmentation[0]


def ga_segmentation(image: str, nb_of_iterations: int, nb_of_clusters: int,
                    coding_type: CodingMethod, selection_method: SelectionMethod,
                    mutation_type: MutationType, crossover_rate: float) -> Tuple:
    """Main algorithm function

    :param image: path to image to be segmented
    :param nb_of_iterations: max number of algorithm iterations
    :param nb_of_clusters: number of groups to divide picture
    :param coding_type: coding method to be used
    :param selection_method: selection method to be used
    :param mutation_type: mutation type to be used
    :param crossover_rate:
    :return: segmented image
    """
    # Read image if exists
    if not check_if_file_exists(image):
        raise FileNotFoundError(f'File {image} does not exist.')

    image_rgb = cv2.imread(image)
    image_rgb = cv2.resize(image_rgb, args.image_size)
    image_size = image_rgb.shape[0:2]

    # According to paper population is initialized randomly considering values <nb_of_clusters-3, nb_of_clusters+3>
    population_size = np.random.randint(nb_of_clusters-3, nb_of_clusters+3)
    # Generate initial population
    population_init = generate_initial_population(population_size=population_size,
                                                  number_of_clusters=nb_of_clusters,
                                                  image_size=image_size,
                                                  image=image_rgb)
    population = population_init
    for iteration in range(nb_of_iterations):
        # According to paper population is initialized randomly considering values <nb_of_clusters-3, nb_of_clusters+3>
        # population_size = np.random.randint(nb_of_clusters - 3, nb_of_clusters + 3)
        # Calculate quality for each element in population
        qualities = []
        for chromosome in population:
            quality = calculate_fitness_function(chromosome=chromosome,
                                                 image_size=image_size,
                                                 image=image_rgb)
            qualities.append(quality)
        # Choose group to create new population
        reproductive_group = select_reproductive_group(current_population=population,
                                                       qualities=qualities,
                                                       method=selection_method)
        # Generate new population based on reproductive group
        new_population = generate_new_population(reproductive_group=reproductive_group,
                                                 qualities=qualities,
                                                 population_size=population_size,
                                                 method=coding_type,
                                                 number_of_clusters=nb_of_clusters,
                                                 image_size=image_size,
                                                 image=image_rgb)
        # Mutate
        new_population_mutated = perform_mutation(new_population=new_population,
                                                  method=mutation_type)
        criterion_ok = check_end_criterion()
        if criterion_ok:
            break
        else:
            population = new_population_mutated
    segmented_image = get_final_result(population=population,
                                       image_size=image_size,
                                       image=image_rgb)
    return segmented_image, image_rgb


if __name__ == '__main__':
    args = parser.parse_args()

    best, orginal = ga_segmentation(image=args.image_path,
                                    nb_of_iterations=args.nb_of_iterations,
                                    nb_of_clusters=args.nb_of_clusters,
                                    coding_type=args.coding_method,
                                    selection_method=args.selection_method,
                                    mutation_type=args.mutation_type,
                                    crossover_rate=args.crossover_rate)

    visualize_based_on_label(labels=best,
                             orginal=orginal)
