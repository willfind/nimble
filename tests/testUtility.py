"""
Tests for nimble._utility submodule
"""
import importlib

import pytest 
import numpy as np
from packaging.version import parse, Version

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import PackageException
from nimble._utility import DeferredModuleImport
from nimble._utility import mergeArguments
from nimble._utility import inspectArguments
from nimble._utility import storm_tuner, hyperopt
from nimble._utility import numpy2DArray, is2DArray
from nimble._utility import _setAll
from nimble._utility import _customMlGetattrHelper
from nimble._dependencies import DEPENDENCIES
from tests.helpers import raises, noLogEntryExpected

def skipIfNoStormTuner():
    if not storm_tuner.nimbleAccessible():
        pytest.skip('Storm Tuner unavailable.')

def skipIfNoHyperOpt():
    if not hyperopt.nimbleAccessible():
        pytest.skip('hyperopt unavailable.')

def test_DeferredModuleImport_numpy():
    optNumpy = DeferredModuleImport('numpy')
    assert optNumpy.nimbleAccessible() # performs the import
    # use some top level numpy functions
    arr_data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    opt_arr = optNumpy.array(arr_data)
    concat = optNumpy.concatenate([opt_arr, opt_arr])

    # use some numpy submodule functions
    randInt = optNumpy.random.randint(1, 10)
    determinant = optNumpy.linalg.det(opt_arr)
    inverse = optNumpy.linalg.inv(opt_arr)

    assert isinstance(randInt, int)
    assert 1 <= randInt < 10
    assert determinant == 1

    import numpy
    import numpy as np
    assert numpy == np
    assert numpy != optNumpy

    numpy_arr = np.array(arr_data)
    assert np.array_equal(numpy_arr, opt_arr)
    assert type(numpy_arr) == type(opt_arr)
    assert np.array_equal(inverse, opt_arr)


def test_DeferredModuleImport_numpy_nimbleAccessibleSuccess():
    optNumpy = DeferredModuleImport('numpy')
    if not optNumpy.nimbleAccessible():
        assert False


@raises(AttributeError)
def test_DeferredModuleImport_numpy_noAccessibleCheck():
    np = DeferredModuleImport('numpy')
    # random is a valid numpy module but will get an AttributeError since
    # we never verified numpy is accessible using np.nimbleAccessible()
    np.random


@raises(AttributeError)
def test_DeferredModuleImport_numpy_bogusAttribute():
    optNumpy = DeferredModuleImport('numpy')
    assert optNumpy.nimbleAccessible() # performs the import
    assert hasattr(optNumpy, 'random')
    optNumpy.random.bogus


def test_DeferredModuleImport_bogus_nimbleAccessibleFailure():
    bogus = DeferredModuleImport('bogus')
    if bogus.nimbleAccessible():
        assert False

def test_DeferredModuleImport_invalidVersion():
    skipIfNoStormTuner()
    skipIfNoHyperOpt()

    opt = []
    for dependency in DEPENDENCIES.values():
        if dependency.section not in ['required', 'interfaces', 'development']:
            opt.append(dependency.name)
    msg = 'does not meet the version requirements'
    for pkg in opt:
        mod = importlib.import_module(pkg)
        if not hasattr(mod, '__version__'):
            continue
        saved = mod.__version__
        try:
            mod.__version__ = '0.0.0'
            assert isinstance(parse(mod.__version__), Version)
            defer = DeferredModuleImport(pkg)
            with raises(PackageException, match=msg):
                defer.nimbleAccessible()
        finally:
            mod.__version__ = saved

@raises(InvalidArgumentValueCombination)
def testMergeArgumentsException():
    """ Test utility.mergeArguments will throw the exception it should """
    args = {1: 'a', 2: 'b', 3: 'd'}
    kwargs = {1: 1, 2: 'b'}

    mergeArguments(args, kwargs)


def testMergeArgumentsHand():
    """ Test utility.mergeArguments is correct on hand construsted data """
    args = {1: 'a', 2: 'b', 3: 'd'}
    kwargs = {1: 'a', 4: 'b'}

    ret = mergeArguments(args, kwargs)

    assert ret == {1: 'a', 2: 'b', 3: 'd', 4: 'b'}


def testMergeArgumentsMakesCopy():
    """Test utility.mergeArguments does not modify or return either original dict"""
    emptyA = {}
    emptyB = {}
    dictA = {'foo': 'bar'}
    dictB = {'bar': 'foo'}

    merged0 = mergeArguments(None, emptyB)
    assert emptyB == merged0
    assert id(emptyB) != id(merged0)
    assert emptyB == {}

    merged1 = mergeArguments(emptyA, emptyB)
    assert emptyA == merged1
    assert emptyB == merged1
    assert id(emptyA) != id(merged1)
    assert id(emptyB) != id(merged1)
    assert emptyA == {}
    assert emptyB == {}

    merged2 = mergeArguments(dictA, emptyB)
    assert dictA == merged2
    assert id(dictA) != id(merged2)
    assert dictA == {'foo': 'bar'}
    assert emptyB == {}

    merged3 = mergeArguments(emptyA, dictB)
    assert dictB == merged3
    assert id(dictB) != id(merged3)
    assert emptyA == {}
    assert dictB == {'bar': 'foo'}

    merged4 = mergeArguments(dictA, dictB)
    assert dictA != merged4
    assert dictB != merged4
    assert dictA == {'foo': 'bar'}
    assert dictB == {'bar': 'foo'}


def test_inspectArguments():

    def checkSignature(a, b, c, d=False, e=True, f=None, *sigArgs, **sigKwargs):
        pass

    a, v, k, d = inspectArguments(checkSignature)

    assert a == ['a', 'b', 'c', 'd', 'e', 'f']
    assert v == 'sigArgs'
    assert k == 'sigKwargs'
    assert d == (False, True, None)

def test_numpy2DArray_converts1D():
    raw = [1, 2, 3, 4]
    ret = numpy2DArray(raw)
    fromNumpy = np.array(raw)
    assert len(ret.shape) == 2
    assert not np.array_equal(ret,fromNumpy)

@raises(InvalidArgumentValue)
def test_numpy2DArray_dimensionException():
    raw = [[[1, 2], [3, 4]]]
    ret = numpy2DArray(raw)

def test_is2DArray():
    raw1D = [1, 2, 3]
    arr1D = np.array(raw1D)
    mat1D = np.matrix(raw1D)
    assert not is2DArray(arr1D)
    assert is2DArray(mat1D)
    raw2D = [[1, 2, 3]]
    arr2D = np.array(raw2D)
    mat2D = np.matrix(raw2D)
    assert is2DArray(arr2D)
    assert is2DArray(mat2D)
    raw3D = [[[1, 2, 3]]]
    arr3D = np.array(raw3D)
    assert not is2DArray(arr3D)

def test_setAll():
    import sys as _sys # always ignored module
    from os import path # module
    from re import search, match, findall, compile # functions

    _variable = '_variable' # always ignored local variable
    variable = 'variable'

    variables = vars().copy()

    all1 = _setAll(variables)
    assert all1 == ['compile', 'findall', 'match', 'search', 'variable']

    all2 = _setAll(variables, includeModules=True)
    assert all2 == ['compile', 'findall', 'match', 'path', 'search',
                    'variable']

    all3 = _setAll(variables, ignore=['match'])
    assert all3 == ['compile', 'findall', 'search', 'variable']

    # ignore takes precedence over includeModules
    all4 = _setAll(variables, ignore=['variable', 'path'],
                   includeModules=True)
    assert all4 == ['compile', 'findall', 'match', 'search']


###############
# __getattr__ #
###############

@noLogEntryExpected
def test_ml_getattr_suggestions():
    data = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    ftNames = ['a', 'b' ,'c', 'label']
    trainData = nimble.data(data, featureNames=ftNames, useLog=False)
    toTest = nimble.train('nimble.KNNClassifier', trainX=trainData,trainY='label', useLog=False)

    try:
        toTest.fit()
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'TrainedLearner' has no attribute 'fit'. Try nimble.train() "
               "or the TrainedLearner's .retrain() method instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("fit") in err_message

    try:
        toTest.fit_transform()
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'TrainedLearner' has no attribute 'fit_transform'. Try "
               "nimble.normalize or a data object's points/features.fillMatching() "
               "and features.normalize() methods instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("fit_transform") in err_message

    try:
        toTest.transform()
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'TrainedLearner' has no attribute 'transform'. Try "
               "nimble.normalize or a data object's points/features.fillMatching() "
               "and features.normalize() methods instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("transform") in err_message

    try:
        toTest.predict()
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'TrainedLearner' has no attribute 'predict'. Try the "
               "TrainedLearner's .apply() method instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("predict") in err_message

    try:
        toTest.score()
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'TrainedLearner' has no attribute 'score'. Try the "
               "TrainedLearner's .getScores() method instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("score") in err_message

    try:
        toTest.get_params()
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'TrainedLearner' has no attribute 'get_params'. "
               "Try nimble.learnerParameters() or the TrainedLearner's "
               ".getAttributes() method instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("get_params") in err_message

@noLogEntryExpected
def test_nimble_ml_getattr_suggestions():
    data = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    ftNames = ['a', 'b' ,'c', 'label']
    trainData = nimble.data(data, featureNames=ftNames, useLog=False)

    try:
        nimble.fit(trainData)
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'nimble' has no attribute 'fit'. Try nimble.train() "
               "or the TrainedLearner's .retrain() method instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("fit") in err_message

    try:
        nimble.fit_transform(trainData)
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'nimble' has no attribute 'fit_transform'. Try "
               "nimble.normalize or a data object's points/features.fillMatching() "
               "and features.normalize() methods instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("fit_transform") in err_message

    try:
        nimble.transform(trainData)
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'nimble' has no attribute 'transform'. Try "
               "nimble.normalize or a data object's points/features.fillMatching() "
               "and features.normalize() methods instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("transform") in err_message

    try:
        nimble.predict(trainData)
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'nimble' has no attribute 'predict'. Try the "
               "TrainedLearner's .apply() method instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("predict") in err_message

    try:
        nimble.score(trainData)
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'nimble' has no attribute 'score'. Try the "
               "TrainedLearner's .getScores() method instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("score") in err_message

    try:
        nimble.get_params(trainData)
    except AttributeError as e:
        err_message = str(e)
        exp = ("module 'nimble' has no attribute 'get_params'. "
               "Try nimble.learnerParameters() or the TrainedLearner's "
               ".getAttributes() method instead.")
        assert err_message == exp
        assert _customMlGetattrHelper("get_params") in err_message
