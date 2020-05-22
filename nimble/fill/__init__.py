"""
Collection of functions primarily providing various methods for filling
values in the data with other values.
"""

from .fill import constant
from .fill import mean
from .fill import median
from .fill import mode
from .fill import forwardFill
from .fill import backwardFill
from .fill import interpolate

__all__ = ['backwardFill', 'constant', 'forwardFill', 'interpolate', 'mean',
           'median', 'mode',]
