"""
Contains the hierarchy of objects which can be used to store data
in UML.

Object creation is meant to be done via UML top level functions such
as UML.createData, even though the objects themselves are contained
within this module. They are avilable only for the purposes of instance
checking, and are excluded from __all__ and the automatically generated
documentation.
"""

from __future__ import absolute_import
from .base import Base
from .base_view import BaseView
from .list import List
from .matrix import Matrix
from .sparse import Sparse
from .dataframe import DataFrame
from .axis import Axis
from .elements import Elements


# We want this for unit testing
available = ['List', 'Matrix', 'Sparse', 'DataFrame']
"""
List of type strings for the concrete objects which subclass
UML.data.Base These may be used in calls to UML.createData or other
similiar object creation methods.
"""

__all__ = ['available', 'Axis', 'Base', 'BaseView', 'Elements']
