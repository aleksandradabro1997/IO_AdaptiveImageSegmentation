import numpy as np

from typing import List
from collections import defaultdict


def generate_new_population_classic(reproductive_group: List, coding_probability: float = 0.75) -> List:
    """Generate new population with classic method

    :param coding_probability: threshold probability for performing coding
    :param reproductive_group: best selected solutions
    :return: new population
    """
    new_population = []
    # Generate pairs for coding
    pairs = np.random.randint(0, len(reproductive_group), (len(reproductive_group), 2))
    for pair in pairs:
        if np.random.randint(0, 100) < 100 * coding_probability:
            # Perform coding
            idx = np.random.randint(0, len(reproductive_group[pair[0]]))
            child = defaultdict()
            for i in range(len(reproductive_group[pair[0]])):
                if i < idx:
                    child[i] = reproductive_group[pair[0]][i]
                else:
                    child[i] = reproductive_group[pair[1]][i]
        # Add parents to population
        new_population.append(reproductive_group[pair[0]])
        new_population.append(reproductive_group[pair[1]])

    return new_population


def generate_new_population_woody(reproductive_group: List) -> List:
    """Generate new population with woody method

    :param reproductive_group:
    :return:
    """
    pass


def generate_new_population_permutational(reproductive_group: List) -> List:
    """Generate new population with permutational method

    :param reproductive_group:
    :return:
    """
    pass