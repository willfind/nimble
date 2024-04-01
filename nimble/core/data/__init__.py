
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Contains the hierarchy of objects which can be used to store data
in nimble.

Object creation is meant to be done via nimble top level functions such
as nimble.data, even though the objects themselves are contained
within this module. They are available only for the purposes of instance
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

# Store matplotlib.pyplot figure numbers by name. Multiple plots can be
# placed on the same figure by indicating the figure name
_plotFigures = {}
