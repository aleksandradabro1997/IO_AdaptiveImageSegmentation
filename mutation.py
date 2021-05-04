import copy
import numpy as np
from typing import List


def perform_mutation_inversion(population: List) -> List:
    """

    :param population: current population
    :return: mutated population
    """
    idx1 = np.random.randint(0, len(population))
    idx2 = np.random.randint(0, len(population))
    if idx1 != idx2:
        tmp = copy.deepcopy(population[idx1])
        population[idx1] = copy.deepcopy(population[idx2])
        population[idx2] = tmp
    return population


def perform_mutation_substitution(population: List) -> List:
    """

    :param population: current population
    :return: mutated population
    """
    pass


def perform_mutation_removal(population: List) -> List:
    """
    Remove chromosome from population
    :param population: current population
    :return: current population
    """
    idx = np.random.randint(0, len(population))
    del population[idx]
    return population


