"""
Contains the hierarchy of objects which can be used to store data
in nimble.

Object creation is meant to be done via nimble top level functions such
as nimble.data, even though the objects themselves are contained
within this module. They are avilable only for the purposes of instance
checking, and are excluded from __all__ and the automatically generated
documentation.
"""

from .base import Base
from .views import BaseView
from .list import List
from .matrix import Matrix
from .sparse import Sparse
from .dataframe import DataFrame
from .axis import Axis
from .points import Points
from .features import Features

# List of type strings for the concrete objects which subclass
# nimble.core.data.Base These may be used in calls to nimble.data or
# other similiar object creation methods.
# We want this for unit testing
available = ['List', 'Matrix', 'Sparse', 'DataFrame']
