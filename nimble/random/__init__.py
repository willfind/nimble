
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
Module for random operations with Nimble.

Nimble is able to guarantee reproduciblility through ``pythonRandom``
and ``numpyRandom``. These are, respectively, instances of Python and
Numpy's randomness objects and are always instantiated using the same
seed value on import. Nimble always uses these instances internally to
maintain the guarantee of reproduciblility. Users wanting to maintain
reproduciblility must also use these instances for random operations.
"""

from .randomness import data
from .randomness import numpyRandom
from .randomness import pythonRandom
from .randomness import setSeed
from .randomness import alternateControl
from .randomness import generateSubsidiarySeed
from .randomness import _getValidSeed
from .._utility import _setAll

# for Sphinx docstring
#: The random.Random instance used within Nimble.
#:
#: This instance must be used to ensure reproduciblility when using
#: Python's randomness.
#:
#: See Also
#: --------
#: numpyRandom, random.Random
#:
#: Keywords
#: --------
#: state, seed, number generator
pythonRandom = pythonRandom # pylint: disable=self-assigning-variable

# for Sphinx docstring
#: The np.random.RandomState instance used within Nimble.
#:
#: This instance must be used to ensure reproduciblility when using
#: Numpy's randomness.
#:
#: See Also
#: --------
#: pythonRandom, numpy.random.RandomState
#:
#: Keywords
#: --------
#: state, seed, number generator
numpyRandom = numpyRandom # pylint: disable=self-assigning-variable



__all__ = _setAll(vars())
