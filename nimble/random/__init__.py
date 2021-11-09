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
from .randomness import _generateSubsidiarySeed
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
pythonRandom = pythonRandom

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
numpyRandom = numpyRandom



__all__ = _setAll(vars())
