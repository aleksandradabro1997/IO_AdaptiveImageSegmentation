from enum import IntEnum


class SelectionMethod(IntEnum):
    """
    Class with selection types for GA.

    - ROULETTE - lets imagine roulette, each element in population gets area proportionally to quality function,
                 then run roulette and take choosen element
    - RANK - sort elements according to quality function and select x best
    - TOURNAMENT - group elements and take best from those groups based on quality function
    """
    ROULETTE = 0
    RANK = 1
    TOURNAMENT = 2


class MutationType(IntEnum):
    """
    Class with mutation types for GA.

    - INVERSION -
    - SUBSTITUTION -
    - REMOVAL -
    """
    INVERSION = 0
    SUBSTITUTION = 1
    REMOVAL = 2


class CodingMethod(IntEnum):
    """
    Class with coding methods for GA.

    - CLASSIC -
    - PERMUTATIONAL -
    - WOODY -
    """
    CLASSIC = 0
    PERMUTATIONAL = 1
    WOODY = 2
