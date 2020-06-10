"""
Collection of functions determining if a data value or entire data
object satisfy a given condition.
"""

from .match import _convertMatchToFunction
from .match import missing
from .match import numeric
from .match import nonNumeric
from .match import zero
from .match import nonZero
from .match import positive
from .match import negative
from .match import infinity
from .match import boolean
from .match import true
from .match import false
from .match import anyValues
from .match import allValues
from .match import anyMissing
from .match import allMissing
from .match import anyNumeric
from .match import allNumeric
from .match import anyNonNumeric
from .match import allNonNumeric
from .match import anyZero
from .match import allZero
from .match import anyNonZero
from .match import allNonZero
from .match import anyPositive
from .match import allPositive
from .match import anyNegative
from .match import allNegative
from .match import anyInfinity
from .match import allInfinity
from .match import anyBoolean
from .match import allBoolean
from .match import anyTrue
from .match import allTrue
from .match import anyFalse
from .match import allFalse


__all__ = ['allBoolean', 'allFalse', 'allInfinity', 'allMissing',
           'allNegative', 'allNonNumeric', 'allNonZero', 'allNumeric',
           'allPositive', 'allTrue', 'allValues', 'allZero', 'anyBoolean',
           'anyFalse', 'anyInfinity', 'anyMissing', 'anyNegative',
           'anyNonNumeric', 'anyNonZero', 'anyNumeric', 'anyPositive',
           'anyTrue', 'anyValues', 'anyZero', 'boolean', 'false', 'infinity',
           'missing', 'negative', 'nonNumeric', 'nonZero', 'numeric',
           'positive', 'true', 'zero']
