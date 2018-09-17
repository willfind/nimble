"""
TODO
"""

from __future__ import absolute_import

from .fill import factory
from .fill import mean
from .fill import median
from .fill import mode
from .fill import forward
from .fill import backward
from .fill import interpolate

__all__ = ['backward', 'factory', 'forward', 'interpolate', 'mean', 'median',
           'mode',]
