"""
Collection of functions providing methods for filling values in the data
with other values.
"""

from .fill import constant
from .fill import mean
from .fill import median
from .fill import mode
from .fill import forwardFill
from .fill import backwardFill
from .fill import interpolate
from .._utility import _setAll

__all__ = _setAll(vars())
