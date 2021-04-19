import copy
import numpy as np
from typing import List


def select_by_rank_method(population: List, qualities: List, reproduction_size: int) -> List:
    """Select reproductive group with ranking method

    :param reproduction_size: size of reproduction group
    :param population: current population
    :param qualities: values of fitness function
    :return: reproductive group - chromosomes to create new population
    """
    # Sort according to quality
    population_sorted = [x for x, _ in sorted(zip(population, qualities), key=lambda pair: pair[1])]
    # Return n best results
    return population_sorted[0:reproduction_size]


def select_by_roulette_method(population: List, qualities: List, reproduction_size: int) -> List:
    """Select reproductive group with roulette method

    :param reproduction_size: size of reproduction group
    :param population: current population
    :param qualities: values of fitness function
    :return: reproductive group - chromosomes to create new population
    """
    pass


def select_by_tournament_method(population: List, qualities: List, reproduction_size: int) -> List:
    """Select reproductive group with tournament method

    :param reproduction_size: size of reproduction group
    :param population: current population
    :param qualities: values of fitness function
    :return: reproductive group - chromosomes to create new population
    """
    reproductive_group = []

    if reproduction_size > len(population)/2:
        missing = reproduction_size - (len(population)/2)
        # Select indexes that will be passed without 'fight'
        missing_idx = np.random.randint(0, len(population), missing)
        population_new = [copy.deepcopy(population[x]) for x in range(0, len(population)) if x not in missing_idx]
        for idx in missing_idx:
            reproductive_group.append(population[idx])
        # For the rest calculate pairs
        nb_of_pairs = reproduction_size - missing
        pairs = np.random.randint(0, len(population_new), (nb_of_pairs, 2))
        for pair in pairs:
            if qualities[pair[0]] > qualities[pair[1]]:
                reproductive_group.append(population[pair[0]])
            else:
                reproductive_group.append(population[pair[1]])
    else:
        # Select pairs randomly
        pairs = np.random.randint(0, len(population), (reproduction_size, 2))
        for pair in pairs:
            if qualities[pair[0]] > qualities[pair[1]]:
                reproductive_group.append(population[pair[0]])
            else:
                reproductive_group.append(population[pair[1]])

    return reproductive_group
