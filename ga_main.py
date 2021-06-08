import cv2
import tqdm
import logging
import argparse
import numpy as np
from typing import Tuple

from enums import SelectionMethod, MutationType, CodingMethod, PopulationSize
from utils import save_run_parameters, plot_to_pdf, print_results, print_results_are
from process_dataset_voc import get_images_and_gt_from_dataset
from comparison import run_slic_algorithm, compare_results_slic, run_k_means
from error_metrics import compare_results, calculate_adapted_rand_error, calculate_adapted_rand_error_slic
from internals import generate_initial_population, calculate_fitness_function, select_reproductive_group, \
    generate_new_population, perform_mutation, get_final_result, check_end_criterion

parser = argparse.ArgumentParser(description='Adaptive Image Segmentation Algorithm')
parser.add_argument('--nb-of-iterations', default=10, type=int,
                    help='Number of algorithms iterations')
parser.add_argument('--coding-method', default=CodingMethod.PUBLICATION, type=int,
                    help='One of 4 coding types (classic, permutational, woody, publication)')
parser.add_argument('--selection-method', default=SelectionMethod.PUBLICATION, type=int,
                    help='Type of selection method (roulette, rank, tournament, publication)')
parser.add_argument('--mutation-type', default=MutationType.PUBLICATION, type=int,
                    help='Mutation type')
parser.add_argument('--crossover-rate', default=0.01, type=float,
                    help='Value of crossover rate')
parser.add_argument('--population-size-determination', default=PopulationSize.PUBLICATION, type=float,
                    help='Way of determining number of clusters')
parser.add_argument('--image-size', default=(200, 200), type=tuple,
                    help='Size of the image to be segmented')
parser.add_argument('--dataset', default='images/simple', type=str,
                    help='Path to dataset to evaluate algorithm')
parser.add_argument('--compare-slic', default=False, type=bool,
                    help='Compare outputs with outputs from SLIC algorithm')
parser.add_argument('--compare-kmeans', default=False, type=bool,
                    help='Compare outputs with outputs from K-MEANS algorithm')
parser.add_argument('--calculate-are', default=True, type=bool,
                    help='Calculate adaptive rand error ')
parser.add_argument('--remove-noise', default=False, type=bool,
                    help='Remove noises or not')


# Set random seed
np.random.seed(seed=1)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logger.addHandler(logging.StreamHandler())


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
    :param crossover_rate: probability of performing crossover
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
    if args.population_size_determination == PopulationSize.PUBLICATION:
        if nb_of_clusters-3 > 0:
            population_size = np.random.randint(nb_of_clusters-3, nb_of_clusters+3)
        else:
            population_size = np.random.randint(1, nb_of_clusters + 3)
    elif args.population_size_determination == PopulationSize.VOC:
        population_size = nb_of_clusters  # add one for background
    else:
        raise ValueError('Invalid method of choosing population size!')
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
            qualities.append((chromosome, quality))
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
                                                 coding_probability=crossover_rate)
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


def ga_main() -> None:
    """Process whole dataset, calculate scores and save results.

    :return: None
    """
    dataset_images, dataset_gt = get_images_and_gt_from_dataset(args.dataset, args.image_size)
    algo_output = {}
    if args.compare_slic:
        slic_output = {}
    if args.compare_kmeans:
        kmeans_output = {}
    qualities_and_populations = {}
    for name, image in dataset_images.items():
        if args.remove_noise:
            image = cv2.fastNlMeansDenoisingColored(image)
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
        if args.compare_slic:
            slic_img = run_slic_algorithm(image=image,
                                          nb_of_clusters=dataset_gt[name][1]+1,
                                          image_size=args.image_size,
                                          nb_of_iterations=args.nb_of_iterations)
            slic_output.update({name: slic_img})

        if args.compare_kmeans:
            kmeans_img = run_k_means(image=image,
                                     nb_of_clusters=dataset_gt[name][1],
                                     image_size=args.image_size,
                                     nb_of_iterations=args.nb_of_iterations)
            kmeans_output.update({name: kmeans_img})

    logging.info('Calculating accuracy scores ... ')
    # Pixel to pixel comparison
    scores = compare_results(ground_truth=dataset_gt,
                             algo_output=algo_output,
                             image_size=args.image_size,
                             images=dataset_images)
    if args.compare_slic:
        scores_slic = compare_results_slic(slic_output, dataset_gt)
    if args.compare_kmeans:
        scores_kmeans = compare_results_slic(kmeans_output, dataset_gt)

    # Adapted Rand Error
    if args.calculate_are:
        scores_are = calculate_adapted_rand_error(dataset_gt, algo_output)
        if args.compare_slic:
            scores_are_slice = calculate_adapted_rand_error_slic(dataset_gt, slic_output)
        if args.compare_kmeans:
            scores_are_kmeans = calculate_adapted_rand_error_slic(dataset_gt, kmeans_output)

    # Print results
    print_results(scores)
    if args.calculate_are:
        print_results_are(scores_are)

    if args.compare_slic:
        logging.info('--------------------SLIC SCORES--------------------')
        print_results(scores_slic)
        if args.calculate_are:
            print_results_are(scores_are_slice)

    if args.compare_kmeans:
        logging.info('--------------------KMEANS SCORES--------------------')
        print_results(scores_kmeans)
        if args.calculate_are:
            print_results_are(scores_are_kmeans)

    logging.info('Saving to pdf ...')
    pdf_name = plot_to_pdf(gt=dataset_gt,
                           algo_output=algo_output,
                           qualities=qualities_and_populations,
                           slice_output=slic_output if args.compare_slic else None,
                           kmeans_output=kmeans_output if args.compare_kmeans else None)

    logging.info('Saving run parameters and results ...')
    save_run_parameters(dataset=args.dataset,
                        image_size=args.image_size,
                        number_of_iterations=args.nb_of_iterations,
                        coding_method=args.coding_method,
                        selection_method=args.selection_method,
                        mutation_type=args.mutation_type,
                        crossover_rate=args.crossover_rate,
                        scores=scores,
                        mean_accuracy=np.mean(list(scores.values())),
                        are_mean=[sum(y) / len(y) for y in zip(*tuple(scores_are.values()))] if args.calculate_are else None,
                        scores_are=scores_are if args.calculate_are else None,
                        scores_slic=scores_slic if args.compare_slic else None,
                        scores_are_slice=scores_are_slice if (args.calculate_are and args.compare_slic) else None,
                        scores_kmeans=scores_kmeans if args.compare_kmeans else None,
                        scores_are_kmeans=scores_are_kmeans if (args.calculate_are and args.compare_kmeans) else None,
                        pdf_file=pdf_name,
                        )


if __name__ == '__main__':
    args = parser.parse_args()
    ga_main()
