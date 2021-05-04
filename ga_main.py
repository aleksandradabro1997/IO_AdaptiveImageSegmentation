import cv2
import tqdm
import logging
import argparse
import numpy as np

from scipy.spatial import distance
from collections import defaultdict
from typing import List, Tuple, Dict
from enums import SelectionMethod, MutationType, CodingMethod
from utils import save_run_parameters, plot_to_pdf, print_results
from selection import select_by_rank_method, select_by_roulette_method, select_by_tournament_method
from coding import generate_new_population_classic, generate_new_population_permutational, generate_new_population_woody
from mutation import perform_mutation_inversion, perform_mutation_removal, perform_mutation_substitution
from process_dataset_voc import get_images_and_gt_from_dataset

parser = argparse.ArgumentParser(description='Adaptive Image Segmentation Algorithm')
parser.add_argument('--nb-of-iterations', default=50, type=int,
                    help='Number of algorithms iterations')
parser.add_argument('--nb-of-clusters', default=4, type=int,
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
parser.add_argument('--dataset', default='images/simple', type=str,
                    help='Path to dataset to evaluate algorithm')

# Set random seed
np.random.seed(seed=1)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


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
    # Buffers for data over iterations
    populations_and_qualities = []

    # Read image if exists
    #if not check_if_file_exists(image):
    #    raise FileNotFoundError(f'File {image} does not exist.')

    image_rgb = image #cv2.imread(image)
    image_rgb = cv2.resize(image_rgb, args.image_size)
    image_size = image_rgb.shape[0:2]

    # According to paper population is initialized randomly considering values <nb_of_clusters-3, nb_of_clusters+3>
    #population_size = np.random.randint(nb_of_clusters-3, nb_of_clusters+3)
    population_size = nb_of_clusters  # add one for background
    # Generate initial population
    population_init = generate_initial_population(population_size=population_size,
                                                  number_of_clusters=nb_of_clusters,
                                                  image_size=image_size,
                                                  image=image_rgb)
    population = population_init
    for iteration in tqdm.tqdm(range(nb_of_iterations)):
        # According to paper population is initialized randomly considering values <nb_of_clusters-3, nb_of_clusters+3>
        # population_size = np.random.randint(nb_of_clusters - 3, nb_of_clusters + 3)
        # Calculate quality for each element in population
        qualities = []
        for chromosome in population:
            quality = calculate_fitness_function(chromosome=chromosome,
                                                 image_size=image_size,
                                                 image=image_rgb)
            qualities.append(quality)
        # Save for later
        populations_and_qualities.append({'population': population,
                                          'quality': qualities})

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
                                                 image=image_rgb,
                                                 current_population=population,
                                                 coding_probability=args.crossover_rate)
        # Mutate
        new_population_mutated = perform_mutation(new_population=new_population,
                                                  method=mutation_type)
        criterion_ok = check_end_criterion()
        if criterion_ok:
            break
        else:
            population = new_population_mutated
    segmented_image, best_chromosome = get_final_result(population=population,
                                                        image_size=image_size,
                                                        image=image_rgb)
    return segmented_image, best_chromosome, populations_and_qualities


def compare_results(ground_truth: Dict, algo_output: Dict):
    """Compare both outputs and return score

    :param ground_truth: Dict with names and gt
    :param algo_output: dict with output
    :return: Dictionary name : score
    """
    scores = {}
    for img_name, gt in ground_truth.items():
        try:
            algo_img = algo_output[img_name][0]
            algo_chromosome = algo_output[img_name][1]
            gt = dataset_gt[name][0]
            best_lab, best_dist = assign_labels_based_on_chromosome(algo_chromosome, args.image_size, image)
            score = 0
            for i in range(dataset_gt[name][1]):
                label_indexes = np.where(algo_img == i)
                centre_index = np.argmin(best_dist[algo_img == 0]) #np.where(best_dist[algo_img == 0] == 0)[0]
                label_gt = gt[label_indexes[0][centre_index], label_indexes[1][centre_index]]
                tmp_gt = np.ones(args.image_size) * 100
                tmp_gt[np.where(gt == label_gt)] = label_gt
                score += np.sum(tmp_gt == algo_img)

            scores.update({img_name: score/(args.image_size[0] * args.image_size[1])})
        except:
            continue
    return scores


if __name__ == '__main__':
    args = parser.parse_args()

    dataset_images, dataset_gt = get_images_and_gt_from_dataset(args.dataset, args.image_size)
    algo_output = {}
    qualities_and_populations = {}
    for name, image in dataset_images.items():
        logging.info(f'Processing image: {name}.png')
        best_img, best_chromosome, qualities = ga_segmentation(image=image,
                                                               nb_of_iterations=args.nb_of_iterations,
                                                               nb_of_clusters=dataset_gt[name][1],
                                                               coding_type=args.coding_method,
                                                               selection_method=args.selection_method,
                                                               mutation_type=args.mutation_type,
                                                               crossover_rate=args.crossover_rate)
        # Save populations and qualities for later
        qualities_and_populations.update({name: qualities})
        algo_output.update({name: (best_img, best_chromosome)})

    logging.info('Calculating accuracy score ... ')
    scores = compare_results(ground_truth=dataset_gt,
                             algo_output=algo_output)
    print_results(scores)

    logging.info('Saving to pdf ...')
    pdf_name = plot_to_pdf(gt=dataset_gt,
                           algo_output=algo_output,
                           qualities=qualities_and_populations)

    logging.info('Saving run parameters and results ...')
    save_run_parameters(dataset=args.dataset,
                        image_size=args.image_size,
                        number_of_iterations=args.nb_of_iterations,
                        coding_method=args.coding_method,
                        selection_method=args.selection_method,
                        mutation_type=args.mutation_type,
                        crossover_rate=args.crossover_rate,
                        scores=scores,
                        pdf_file=pdf_name)

    pass
    #                                nb_of_iterations=args.nb_of_iterations,
    #                                nb_of_clusters=args.nb_of_clusters,
    #                                coding_type=args.coding_method,
    #                                selection_method=args.selection_method,
    #                                mutation_type=args.mutation_type,
    #                                crossover_rate=args.crossover_rate)

    #visualize_based_on_label(labels=best, orginal=orginal)