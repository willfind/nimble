"""
Tests for the nimble.random submodule
"""

import random
import copy

import numpy as np

import nimble
from tests.helpers import oneLogEntryExpected, logCountAssertionFactory


def testSetRandomSeedExplicit():
    """ Test nimble.random.setSeed yields Nimble accessible random objects with the correct random behavior """
    expPy = random.Random(1333)
    expNp = np.random.RandomState(1333)
    nimble.random.setSeed(1333)

    for i in range(50):
        assert nimble.random.pythonRandom.random() == expPy.random()
        assert nimble.random.numpyRandom.rand() == expNp.rand()


def testSetRandomSeedNone():
    """ Test nimble.random.setSeed operates as expected when passed None (-- use system time as seed) """
    nimble.random.setSeed(None)
    pyState = nimble.random.pythonRandom.getstate()
    npState = nimble.random.numpyRandom.get_state()

    origPy = random.Random()
    origPy.setstate(pyState)
    origNp = np.random.RandomState()
    origNp.set_state(npState)

    nimble.random.setSeed(None)

    assert origPy.random() != nimble.random.pythonRandom.random()
    assert origNp.rand() != nimble.random.numpyRandom.rand()


@logCountAssertionFactory(3)
def testSetRandomSeedPropagate():
    """ Test that nimble.random.setSeed will correctly control how randomized methods in nimble perform """
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [17, 18, 19], [2, 2, 2], [3, 3, 3], [4, 4, 4],
            [5, 5, 5]]
    featureNames = ['1', '2', '3']
    toTest1 = nimble.data("List", data, featureNames=featureNames, useLog=False)
    toTest2 = toTest1.copy()
    toTest3 = toTest1.copy()

    nimble.random.setSeed(1337)
    toTest1.points.permute(useLog=False)

    nimble.random.setSeed(1336)
    toTest2.points.permute(useLog=False)

    nimble.random.setSeed(1337)
    toTest3.points.permute(useLog=False)

    assert toTest1 == toTest3
    assert toTest1 != toTest2


###################
### random.data ###
###################

returnTypes = copy.copy(nimble.core.data.available)

def testReturnsFundamentalsCorrect():
    """
    function that tests
    -the size of the underlying data is consistent with that requested through our API
    -the data class requested (Matrix, Sparse, List) is that which you get back
    -the data fundamental data type used to store the value of (point, feature) pairs
        is what the user requests ('int' or 'float')
    Note:
    These tests are run for all combinations of the paramaters:
        supportedFundamentalTypes = ['int', 'float']
        returnTypes = ['Matrix','Sparse','List']
        sparsities = [0.0, 0.5, .99]
    """

    supportedFundamentalTypes = ['int', 'float']
    sparsities = [0.0, 0.5, .99]

    nPoints = 100
    nFeatures = 200
    #sparsity = .5

    expected = {}
    for curType in supportedFundamentalTypes:
        for curReturnType in returnTypes:
            for curSparsity in sparsities:
                nimble.random.setSeed(1)
                returned = nimble.random.data(curReturnType, nPoints, nFeatures,
                                              curSparsity, elementType=curType)

                assert (len(returned.points) == nPoints)
                assert (len(returned.features) == nFeatures)

                # assert that the requested numerical type was returned
                assert type(returned[0, 0] == curType)

                if curType not in expected:
                    expected[curType] = {}
                if curSparsity not in expected[curType]:
                    expected[curType][curSparsity] = returned.copy('pythonlist')
                else:
                    assert returned.copy('pythonlist') == expected[curType][curSparsity]


#note: makes calls to Base._data with assumptions about underlying datatstructure for sparse data
def testSparsityReturnedPlausible():
    """
    function that tests:
    -for a dataset with 1500 points and 2000 features (2M pairs) that the number of
        zero entries is reasonably close to the amount requested.
    Notes:
    -Because the generation of zeros is done stochastically, exact numbers of zeros
        is not informative. Instead, the test checks that the ratio of zeros to all
        points (zeros and non zeros) is within 1 percent of the 1 - sparsity.
    -These tests are run for all combinations of the paramaters:
        supportedFundamentalTypes = ['int', 'float']
        returnTypes = ['Matrix', 'Sparse', 'List', 'DataFrame']
        sparsities = [0.0, 0.5, .99]
    """
    supportedFundamentalTypes = ['int', 'float']
    sparsities = [0.0, 0.5, .99]

    nPoints = 500
    nFeatures = 1000
    #sparsity = .5

    for curType in supportedFundamentalTypes:
        for curReturnType in returnTypes:
            for curSparsity in sparsities:
                returned = nimble.random.data(curReturnType, nPoints, nFeatures,
                                              curSparsity, elementType=curType)

                if curReturnType.lower() == 'sparse':
                    nonZerosCount = returned._data.nnz
                    actualSparsity = 1.0 - nonZerosCount / float(nPoints * nFeatures)
                    difference = abs(actualSparsity - curSparsity)

                    assert (difference < .01)
                else:
                    nonZerosCount = np.count_nonzero(returned.copy(to='numpyarray'))
                    actualSparsity = 1.0 - nonZerosCount / float(nPoints * nFeatures)
                    difference = abs(actualSparsity - curSparsity)

                    assert (difference < .01)


def test_createRandomizedData_names_passed():
    '''
    function that tests:
    - Given correctly sized lists of strings for pointNames and featureNames
    arguments, checks if the returned object has those axis names.

    - Validity checking of pointNames and featureNames is not tested
    since it is done exclusively in nimble.data. We only check for successful
    behavior.

    - These tests are run for all combinations of the paramaters:
    supportedFundamentalTypes = ['int', 'float']
    returnTypes = ['Matrix','Sparse','List']
    sparsities = [0.0, 0.5, .99]
    '''
    supportedFundamentalTypes = ['int', 'float']
    sparsities = [0.0, 0.5, .99]

    numberPoints = 10
    numberFeatures = 3
    pnames = ['p{}'.format(i) for i in range(0, numberPoints)]
    fnames = ['f{}'.format(i) for i in range(0, numberFeatures)]

    # TODO create a function summarizing the calling of the function with
    # the different combinations.
    for curType in supportedFundamentalTypes:
        for curReturnType in returnTypes:
            for curSparsity in sparsities:
                ret = nimble.random.data(
                    curReturnType, numberPoints, numberFeatures, curSparsity,
                    elementType=curType, pointNames=pnames, featureNames=fnames)

                assert ret.points.getNames() == pnames
                assert ret.features.getNames() == fnames

def test_random_data_logCount():

    @oneLogEntryExpected
    def byType(rType):
        toTest = nimble.random.data(rType, 5, 5, 0)

    for t in returnTypes:
        byType(t)

####################
# alternateControl #
####################

def testalternateControlExplicit():
    """ Test nimble.random.alternateControl yields Nimble accessible random objects with the correct random behavior """
    expPy1333 = random.Random(1333)
    expNp1333 = np.random.RandomState(1333)
    expPy1334 = random.Random(1334)
    expNp1334 = np.random.RandomState(1334)

    nimble.random.setSeed(1333)
    for i in range(25):
        assert nimble.random.pythonRandom.random() == expPy1333.random()
        assert nimble.random.numpyRandom.rand() == expNp1333.rand()
    with nimble.random.alternateControl(1334):
        for i in range(50):
            assert nimble.random.pythonRandom.random() == expPy1334.random()
            assert nimble.random.numpyRandom.rand() == expNp1334.rand()
    for i in range(25):
        assert nimble.random.pythonRandom.random() == expPy1333.random()
        assert nimble.random.numpyRandom.rand() == expNp1333.rand()


@logCountAssertionFactory(4)
def testalternateControlNone():
    """ Test nimble.random.alternateControl operates as expected when passed None (-- use system time as seed) """
    with nimble.random.alternateControl(None):
        pyState = nimble.random.pythonRandom.getstate()
        npState = nimble.random.numpyRandom.get_state()

        origPy = random.Random()
        origPy.setstate(pyState)
        origNp = np.random.RandomState()
        origNp.set_state(npState)

    with nimble.random.alternateControl(None):
        assert origPy.random() != nimble.random.pythonRandom.random()
        assert origNp.rand() != nimble.random.numpyRandom.rand()

def testalternateControlNested():
    expPy1333 = random.Random(1333)
    expNp1333 = np.random.RandomState(1333)
    expPy1334 = random.Random(1334)
    expNp1334 = np.random.RandomState(1334)

    with nimble.random.alternateControl(1333):
        for i in range(10):
            assert nimble.random.pythonRandom.random() == expPy1333.random()
            assert nimble.random.numpyRandom.rand() == expNp1333.rand()
        with nimble.random.alternateControl(1334):
            for i in range(10):
                assert nimble.random.pythonRandom.random() == expPy1334.random()
                assert nimble.random.numpyRandom.rand() == expNp1334.rand()
        for i in range(10):
            assert nimble.random.pythonRandom.random() == expPy1333.random()
            assert nimble.random.numpyRandom.rand() == expNp1333.rand()

def testalternateControlSetSeedException():
    from nimble.random import setSeed
    try:
        with nimble.random.alternateControl(1):
            nimble.random.setSeed(2)
        assert False # expected RuntimeError
    except RuntimeError:
        pass

    try:
        with nimble.random.alternateControl(1):
            setSeed(2)
        assert False # expected RuntimeError
    except RuntimeError:
        pass

    try:
        with nimble.random.alternateControl(1):
            with nimble.random.alternateControl():
                nimble.random.setSeed(2)
        assert False # expected RuntimeError
    except RuntimeError:
        pass

    try:
        with nimble.random.alternateControl(1):
            with nimble.random.alternateControl():
                nimble.random.pythonRandom.random()
            setSeed(2)
        assert False # expected RuntimeError
    except RuntimeError:
        pass

#todo check that sizes of returned objects are what you request via npoints and nfeatures
