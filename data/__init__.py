"""
Contains the hierarchy of objects which can be used to store data
in UML.

The base class and the concrete classes are made available here
for purposes of documentation and in-code instance checking.
Actual object creation is meant to be done via UML.createData
and the related functions in top level UML.

"""

from base import Base
from dataHelpers import View
from list import List
from matrix import Matrix
from sparse import Sparse


# We want this for unit testing
available = ['List', 'Matrix', 'Sparse']
__all__ = ['Base', 'List', 'Matrix', 'Sparse', 'View']
