import numpy as np
from typing import Tuple, List
from collections import defaultdict
from scipy.spatial import distance

from enums import SelectionMethod, MutationType, CodingMethod
from selection import select_by_rank_method, select_by_roulette_method, select_by_tournament_method
from coding import generate_new_population_classic, generate_new_population_permutational, generate_new_population_woody
from mutation import perform_mutation_inversion, perform_mutation_removal, perform_mutation_substitution


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
    for i in range(population_size):
        chromosome = defaultdict()
        for j in range(0, number_of_clusters):
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
    # There where centre_distances == 0 set high value - to avoid choosing it
    centre_distances = np.where(centre_distances == 0, 1e100, centre_distances)
    for k in range(len(chromosome)):
        try:
            min_distance = np.amin(centre_distances[labels == k])
        except:
            continue
        quality += min_distance
        # Replace chromosome centre with mean cluster values
        indices = np.where(labels == k)
        new_center = (int(sum(indices[0])/len(indices[0])), int(sum(indices[1])/len(indices[1])))
        chromosome[k] = image[new_center[0], new_center[1], :]
    return quality


def generate_new_population(reproductive_group: List, qualities: List, method: CodingMethod,
                            population_size: int, number_of_clusters: int, image_size: Tuple,
                            image: np.array, coding_probability: float, current_population: List) -> List:
    """Generate new population, based on chromosomes selected from current population,
    according to publication best chromosome survives others are random

    :param current_population: population in current iteration
    :param coding_probability: threshold probability for performing coding
    :param population_size: number of solutions in population
    :param image: input image
    :param number_of_clusters: number of parts to divide picture
    :param image_size: size of the input  image
    :param qualities: values of the fitness function for each solution
    :param reproductive_group: group with best, according to fitness function, solutions
    :param method: the way of coding
    :return: new population
    """
    if method == CodingMethod.CLASSIC:
        new_population = generate_new_population_classic(reproductive_group=reproductive_group,
                                                         coding_probability=coding_probability)
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
                                                   qualities=qualities,
                                                   reproduction_size=int(len(current_population)/2))
    elif method == SelectionMethod.ROULETTE:
        reproductive_group = select_by_roulette_method(population=current_population,
                                                       qualities=qualities,
                                                       reproduction_size=int(len(current_population)/2))
    elif method == SelectionMethod.TOURNAMENT:
        reproductive_group = select_by_tournament_method(population=current_population,
                                                         qualities=qualities,
                                                         reproduction_size=int(len(current_population)/2))
    elif method == SelectionMethod.PUBLICATION:
        reproductive_group = current_population
    else:
        raise ValueError(f'Invalid selection method passed!')
    return reproductive_group


def perform_mutation(new_population: List, method: MutationType, mutation_probability: float = 0.1) -> List:
    """Mutate some of the chromosomes.

    :param mutation_probability: probability of performing mutation
    :param new_population: generated new solutions
    :param method: type of the mutation
    :return: mutated solution
    """
    if np.random.randint(0, 100) < 100 * mutation_probability:
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

    :param image: input image
    :param image_size: size of the input image
    :param population: object
    :return: segmented image.
    """
    qualities = []
    for chromosome in population:
        quality = calculate_fitness_function(chromosome, image_size, image)
        qualities.append(quality)

    best_chromosome = select_best_chromosome(population, qualities)
    best_segmentation = assign_labels_based_on_chromosome(best_chromosome, image_size, image)
    return best_segmentation[0], best_chromosome