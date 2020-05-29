"""
Module defining exceptions used in nimble.
"""

from .exceptions import NimbleException
from .exceptions import InvalidArgumentType
from .exceptions import InvalidArgumentValue
from .exceptions import InvalidArgumentTypeCombination
from .exceptions import InvalidArgumentValueCombination
from .exceptions import ImproperObjectAction
from .exceptions import PackageException
from .exceptions import FileFormatException
from .exceptions import _prettyListString
from .exceptions import _prettyDictString

__all__ = ['NimbleException', 'InvalidArgumentType', 'InvalidArgumentValue',
           'InvalidArgumentTypeCombination', 'InvalidArgumentValueCombination',
           'ImproperObjectAction', 'PackageException', 'FileFormatException']
