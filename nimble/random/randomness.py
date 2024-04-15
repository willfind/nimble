
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
Contains functions and objects controlling how randomness is used in
Nimble functions and tests.
"""

import random
from contextlib import contextmanager

import numpy as np

from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import PackageException
from nimble._utility import scipy
from nimble.core.logger import handleLogging
from nimble.core._createHelpers import validateReturnType, initDataObject

pythonRandom = random.Random(42)
numpyRandom = np.random.RandomState(42) # pylint: disable=no-member

def setSeed(seed, *, useLog=None):
    """
    Set the seeds on all sources of randomness in nimble.

    Parameters
    ----------
    seed : int
        Seed for random state. Must be convertible to 32 bit unsigned
        integer for compliance with numpy. If seed is ``None``, the
        operating system's randomness sources are used if available
        otherwise the system time is used.
    """
    if not setSeed._settable:
        msg = 'The global random seed cannot be set while within an alternate '
        msg += 'state of randomness'
        raise RuntimeError(msg)

    pythonRandom.seed(seed)
    numpyRandom.seed(seed)

    handleLogging(useLog, 'setSeed', 'random.setSeed', seed=seed)

# When using alternateControl, using setSeed is not allowed.
setSeed._settable = True

def data(numPoints, numFeatures, sparsity, pointNames='automatic',
         featureNames='automatic', elementType='float', returnType=None,
         name=None, randomSeed=None, *, useLog=None):
    """
    Generate a data object with random contents.

    The range of values and the distribution are dependent on the
    elementType. For the default elementType "float", the data will be a
    normal distribution of values with mean 0 and standard deviation of
    1. If the elementType is "int", values are sampled from a uniform
    distribution of the range 1 to 100.

    Parameters
    ----------
    numPoints : int
        The number of points in the returned object.
    numFeatures : int
        The number of features in the returned object.
    sparsity : float
        The likelihood that the value of a (point,feature) pair is zero.
    elementType : str
        If 'float' (default) then the value of (point, feature) pairs
        are sampled from a normal distribution (location 0, scale 1). If
        elementType is 'int' then value of (point, feature) pairs are
        sampled from uniform integer distribution [1 100]. Zeros are not
        counted in/do not affect the aforementioned sampling
        distribution.
    pointNames : 'automatic', list, dict
        Names to be associated with the points in the returned object.
        If 'automatic', default names will be generated. Otherwise, may
        be specified explictly by a list-like or dict-like object, so
        long as all points in the data are assigned a name and the names
        for each point are unique.
    featureNames : 'automatic', list, dict
        Names to be associated with the features in the returned object.
        If 'automatic', default names will be generated. Otherwise, may
        be specified explictly by a list-like or dict-like object, so
        long as all features in the data are assigned a name and the
        names for each feature are unique.
    returnType : str, None
        Indicates which Nimble data object to return. Options are the
        **case sensitive** strings "List", "Matrix", "Sparse" and
        "DataFrame". If None, Nimble will detect the most appropriate
        type from the data and/or packages available in the environment.
    name : str
        When not None, this value is set as the name attribute of the
        returned object.
    randomSeed : int
       Provide a randomSeed for generating the random data. When None,
       the randomness is controlled by Nimble's random seed.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.

    Returns
    -------
    nimble.core.data.Base
        Subclass of Base object corresponding with the ``returnType``.

    See Also
    --------
    nimble.data, nimble.ones, nimble.zeros, nimble.identity

    Examples
    --------
    Random integers.

    >>> nimble.random.setSeed(42)
    >>> ptNames = ['a', 'b', 'c', 'd', 'e']
    >>> random = nimble.random.data(5, 5, 0, pointNames=ptNames,
    ...                             elementType='int')
    >>> random
    <Matrix 5pt x 5ft
         0   1   2   3   4
       ┌───────────────────
     a │ 22  54  72  91  21
     b │ 32  19   2  46  49
     c │ 86   1  91  91  69
     d │ 30  44  97  39  63
     e │ 42  92  32  55  65
    >

    Random floats, high sparsity.

    >>> nimble.random.setSeed(42)
    >>> sparse = nimble.random.data(5, 5, .9, returnType="Sparse")
    >>> sparse
    <Sparse 5pt x 5ft
           0      1       2      3       4
       ┌────────────────────────────────────
     0 │ 0.000   0.000  0.000   0.000  0.000
     1 │ 0.000   0.000  0.000  -1.283  0.000
     2 │ 0.000  -0.298  0.000   0.000  0.000
     3 │ 0.000   0.000  0.000   0.000  0.000
     4 │ 0.000   0.000  0.000   0.000  0.000
    >
    """
    validateReturnType(returnType)
    if numPoints < 1:
        msg = "must specify a positive nonzero number of points"
        raise InvalidArgumentValue(msg)
    if numFeatures < 1:
        msg = "must specify a positive nonzero number of features"
        raise InvalidArgumentValue(msg)
    if sparsity < 0 or sparsity >= 1:
        msg = "sparsity must be greater than zero and less than one"
        raise InvalidArgumentType(msg)
    if elementType not in ["int", "float"]:
        raise InvalidArgumentValue("elementType may only be 'int' or 'float'")

    randomSeed = _getValidSeed(randomSeed)
    with alternateControl(seed=randomSeed, useLog=False):
        #note: sparse uses rigid density measures, not stochastic sparsity
        size = (numPoints, numFeatures)
        if sparsity > 0:
            density = 1.0 - float(sparsity)
            gridSize = numPoints * numFeatures
            numNonZeroValues = int(gridSize * density)
            # We want to sample over positions, not point/feature indices, so
            # we consider the possible positions as numbered in a row-major
            # order on a grid, and sample that without replacement
            nzLocation = numpyRandom.choice(gridSize, size=numNonZeroValues,
                                            replace=False)
            if elementType == 'int':
                dataVector = numpyRandom.randint(low=1, high=100,
                                                 size=numNonZeroValues)
                dtype = np.dtype(int)
            else: #numeric type is float; distribution is normal
                dataVector = numpyRandom.normal(0, 1, size=numNonZeroValues)
                dtype = np.dtype(float)
            if returnType == 'Sparse':
                if not scipy.nimbleAccessible():
                    msg = "scipy is not available"
                    raise PackageException(msg)
                # The point value is determined by counting how many groups of
                # numFeatures fit into the position number
                pointIndices = np.floor(nzLocation / numFeatures)
                # The feature value is determined by counting the offset from
                # each point edge.
                featureIndices = nzLocation % numFeatures
                randData = scipy.sparse.coo_matrix(
                    (dataVector, (pointIndices, featureIndices)),
                    (numPoints, numFeatures))
            else:
                randData = np.zeros(size, dtype=dtype)
                # np.put indexes as if flat so can use nzLocation
                np.put(randData, nzLocation, dataVector)
        else:
            if elementType == 'int':
                randData = numpyRandom.randint(1, 100, size=size)
            else:
                randData = numpyRandom.normal(loc=0.0, scale=1.0, size=size)

    ret = initDataObject(randData, pointNames=pointNames,
                         featureNames=featureNames, returnType=returnType,
                         name=name, copyData=False, skipDataProcessing=True)

    handleLogging(useLog, 'load', ret, sparsity=sparsity, seed=randomSeed)

    return ret


def generateSubsidiarySeed():
    """
    Generate a random integer usable as a python or numpy random seed.

    Used for when it is necessary to make a subroutine call or object
    external to Nimble. By generating a seed using Nimble's random state,
    the execution of that external process is made to be contingent on
    Nimble's random state, and should therefore be subject to the same
    reproducibility guarantees.

    See Also
    --------
    pythonRandom, numpyRandom, alternateControl
    """
    # must range from zero to maxSeed because numpy random wants an
    # unsigned 32 bit int. Negative numbers can cause conversion errors,
    # and larger numbers can cause exceptions.
    maxSeed = (2 ** 32) - 1
    return pythonRandom.randint(1, maxSeed)

def _getValidSeed(seed):
    """
    Validate the random seed value.
    """
    if seed is None:
        seed = generateSubsidiarySeed()
    elif not isinstance(seed, int):
        raise InvalidArgumentType('seed must be an integer')
    elif not 0 <= seed <= (2 ** 32) - 1:
        msg = 'randomSeed is required to be an unsigned 32 bit integer. '
        msg += 'Set randomSeed such that 0<=randomSeed<=4294967295.'
        raise InvalidArgumentValue(msg)

    return seed


def _stillDefaultState():
    """
    Determine if the random state is default or has been modified.

    Will return False if setSeed has been called to modify the
    state of the internal sources of randomness (with the exception of
    calls made within a section designated by startUncontrolledSection
    and endUncontrolledSection). Return True if there has been no
    modification during this 'session' of nimble. To be used in unit
    tests which rely on comparing hard coded results to calls to
    functions that rely on randomness in order to abort if we are in an
    unpredictable state.
    """
    # for now, don't know exactly to to make this correct, so anything that
    # uses it should explode
    assert False

#return _stillDefault

@contextmanager
def alternateControl(seed=None, *, useLog=None):
    """
    Context manager to operate outside of the current random state.

    Saves the state prior to entering, changes the state on entry, and
    restores the original state on exit. The setSeed function is
    disallowed while within the context manager as this would modify the
    controlled state. However, these objects can be nested to achieve a
    different state while already within a controlled state.

    Parameters
    ----------
    seed : int
        Seed for random state. Must be convertible to 32 bit unsigned
        integer for compliance with numpy. If seed is ``None``, the
        operating system's randomness sources are used if available
        otherwise the system time is used.
    useLog : bool, None
        Local control for whether to send object creation to the logger.
        If None (default), use the value as specified in the "logger"
        "enabledByDefault" configuration option. If True, send to the
        logger regardless of the global option. If False, do **NOT**
        send to the logger, regardless of the global option.
    """
    pythonState = pythonRandom.getstate()
    numpyState = numpyRandom.get_state()
    savedSettable = setSeed._settable
    try:
        pythonRandom.seed(seed)
        numpyRandom.seed(seed)
        setSeed._settable = False

        action = 'entered random.alternateControl'
        handleLogging(useLog, 'setSeed', action, seed=seed)
        yield
    finally:
        pythonRandom.setstate(pythonState)
        numpyRandom.set_state(numpyState)
        setSeed._settable = savedSettable

        action = 'exited random.alternateControl'
        handleLogging(useLog, 'setSeed', action, seed=seed)
