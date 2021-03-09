"""
Contains functions and objects controlling how randomness is used in
nimble functions and tests.
"""

import random

import numpy

import nimble # pylint: disable=unused-import
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import PackageException
from nimble._utility import scipy
from nimble.core.logger import handleLogging
from nimble.core._createHelpers import validateReturnType, initDataObject

pythonRandom = random.Random(42)
numpyRandom = numpy.random.RandomState(42) # pylint: disable=no-member

class _RandomControl:
    """
    Track the random states controlled by Nimble.
    """
    # We use None to signal that we are within a section of
    # controlled randomness
    pythonState = None
    numpyState = None

def setSeed(seed, useLog=None):
    """
    Set the seeds on all sources of randomness in nimble.

    Parameters
    ----------
    seed : int
        Seed for random state. Must be convertible to 32 bit unsigned
        integer for compliance with numpy. If seed is None, then we use
        os system time.
    """

    pythonRandom.seed(seed)
    numpyRandom.seed(seed)

    handleLogging(useLog, 'setSeed', seed=seed)


def data(
        returnType, numPoints, numFeatures, sparsity, pointNames='automatic',
        featureNames='automatic', elementType='float', name=None,
        randomSeed=None, useLog=None):
    """
    Generate a data object with random contents.

    The range of values is dependent on the elementType.

    Parameters
    ----------
    returnType : str
        May be any of the allowed types specified in
        nimble.core.data.available.
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
    >>> random = nimble.random.data('Matrix', 5, 5, 0, pointNames=ptNames,
    ...                             elementType='int')
    >>> random
    Matrix(
        [[22 54 72 91 21]
         [32 19 2  46 49]
         [86 1  91 91 69]
         [30 44 97 39 63]
         [42 92 32 55 65]]
        pointNames={'a':0, 'b':1, 'c':2, 'd':3, 'e':4}
        )

    Random floats, high sparsity.

    >>> nimble.random.setSeed(42)
    >>> sparse = nimble.random.data('Sparse', 5, 5, .9)
    >>> sparse
    Sparse(
        [[0   0    0   0    0]
         [0   0    0 -1.283 0]
         [0 -0.298 0   0    0]
         [0   0    0   0    0]
         [0   0    0   0    0]]
        )
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
    _startAlternateControl(seed=randomSeed)
    #note: sparse is not stochastic sparsity, it uses rigid density measures
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
        else: #numeric type is float; distribution is normal
            dataVector = numpyRandom.normal(0, 1, size=numNonZeroValues)
        if returnType == 'Sparse':
            if not scipy.nimbleAccessible():
                msg = "scipy is not available"
                raise PackageException(msg)
            # The point value is determined by counting how many groups of
            # numFeatures fit into the position number
            pointIndices = numpy.floor(nzLocation / numFeatures)
            # The feature value is determined by counting the offset from each
            # point edge.
            featureIndices = nzLocation % numFeatures
            randData = scipy.sparse.coo.coo_matrix(
                (dataVector, (pointIndices, featureIndices)),
                (numPoints, numFeatures))
        else:
            randData = numpy.zeros(size)
            # numpy.put indexes as if flat so can use nzLocation
            numpy.put(randData, nzLocation, dataVector)
    else:
        if elementType == 'int':
            randData = numpyRandom.randint(1, 100, size=size)
        else:
            randData = numpyRandom.normal(loc=0.0, scale=1.0, size=size)
    _endAlternateControl()

    handleLogging(useLog, 'load', "Random " + returnType, numPoints,
                  numFeatures, name, sparsity=sparsity, seed=randomSeed)

    return initDataObject(returnType, rawData=randData, pointNames=pointNames,
                          featureNames=featureNames, name=name,
                          skipDataProcessing=True)


def _generateSubsidiarySeed():
    """
    Randomly generate an integer seed.

    The seed will be used in a call to a subroutine our external system,
    so that even though our internal sources of randomness are not used,
    the results are still dependent on our random state.
    """
    # must range from zero to maxSeed because numpy random wants an
    # unsigned 32 bit int. Negative numbers can cause conversion errors,
    # and larger numbers can cause exceptions. 0 has no effect on randomness
    # control in shogun so start at 1.
    maxSeed = (2 ** 32) - 1
    return pythonRandom.randint(1, maxSeed)

def _getValidSeed(seed, forShogun=False):
    """
    Validate the random seed value.
    """
    if seed is None:
        seed = _generateSubsidiarySeed()
    elif not isinstance(seed, int):
        raise InvalidArgumentType('seed must be an integer')
    elif forShogun and seed == 0:
        msg = "The seed 0 does not generate reproducible results in shogun. "
        msg += "Set randomSeed such that 1<=randomSeed<=4294967295."
        raise InvalidArgumentValue(msg)
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

def _startAlternateControl(seed=None):
    """
    Begin using a temporary new seed for randomness.

    Called to open a certain section of code that needs to a different
    kind of randomness than the current default, without changing the
    reproducibility of later random calls outside of the section. This
    saves the state of the nimble internal random sources, and calls
    setSeed using the given parameter. The saved state can then be
    restored with a call to_endAlternateControl. Meant to be used in
    unit tests, to either protect later calls from the modifications in
    this section, or to ensure consistency regardless of the current
    state of randomness in nimble.

    Parameters
    ----------
    seed : int
        Seed for random state. Must be convertible to 32 bit unsigned
        integer for compliance with numpy. If seed is None, then we use
        os system time.
    """
    _RandomControl.pythonState = pythonRandom.getstate()
    _RandomControl.numpyState = numpyRandom.get_state()

    setSeed(seed, useLog=False)


def _endAlternateControl():
    """
    Stop using the temporary seed created by `_startAlternateControl``.

    Called to close a certain section of code that needs to have
    different kind of randomness than the current default without
    changing the reproducibility of later random calls outside of the
    section. This will restore the state saved by
    ``_startAlternateControl``.
    """
    if _RandomControl.pythonState is not None:
        pythonRandom.setstate(_RandomControl.pythonState)
        _RandomControl.pythonState = None
    if _RandomControl.numpyState is not None:
        numpyRandom.set_state(_RandomControl.numpyState)
        _RandomControl.numpyState = None
