"""
Collection of functions primarily determining if a data value or entire
data object satisfy a given condition.
"""
from __future__ import absolute_import

from .match import convertMatchToFunction
from .match import missing
from .match import numeric
from .match import nonNumeric
from .match import zero
from .match import nonZero
from .match import positive
from .match import negative
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

__all__ = ['allValues', 'allMissing', 'allNegative', 'allNonNumeric',
           'allNonZero', 'allNumeric', 'allPositive', 'allZero',
           'anyValues', 'anyMissing', 'anyNegative', 'anyNonNumeric',
           'anyNonZero', 'anyNumeric', 'anyPositive', 'anyZero',
           'convertMatchToFunction', 'missing', 'negative', 'nonNumeric',
           'nonZero', 'numeric', 'positive', 'zero',
           ]
