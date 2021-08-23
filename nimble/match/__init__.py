"""
Collection of functions determining if a data value or entire data
object satisfy a given condition.
"""

from .match import _convertMatchToFunction
from .match import missing
from .match import nonMissing
from .match import numeric
from .match import nonNumeric
from .match import zero
from .match import nonZero
from .match import positive
from .match import negative
from .match import infinity
from .match import boolean
from .match import integer
from .match import floating
from .match import true
from .match import false
from .match import anyValues
from .match import allValues
from .match import anyMissing
from .match import allMissing
from .match import anyNonMissing
from .match import allNonMissing
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
from .match import anyInteger
from .match import allInteger
from .match import anyFloating
from .match import allFloating
from .match import anyTrue
from .match import allTrue
from .match import anyFalse
from .match import allFalse
from .query import QueryString
from .._utility import _setAll

__all__ = _setAll(vars())
