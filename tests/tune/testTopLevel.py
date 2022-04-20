"""
Test Tuning function available at nimble top level.

Includes: Tune, Tuning
"""

import numpy as np

import nimble
from nimble import Tune, Tuning
from nimble.core.tune import KFold, LeaveOneOut, LeaveOneGroupOut
from nimble.core.tune import HoldoutData, HoldoutProportion
from nimble.core.tune import BruteForce, Consecutive, Bayesian, Iterative
from nimble.core.tune import StochasticRandomMutator
from nimble.calculate import fractionIncorrect, meanAbsoluteError
from nimble.exceptions import InvalidArgumentValue, ImproperObjectAction
from nimble.exceptions import InvalidArgumentValueCombination

from tests.helpers import raises, noLogEntryExpected

def test_Tune():
    crossVal = Tune([1, 2, 3])
    assert len(crossVal) == 3
    assert crossVal[1] == 2
    assert str(crossVal) == 'Tune(values=[1, 2, 3])'
    assert repr(crossVal) == "Tune(values=[1, 2, 3])"

def test_Tune_exceptions():
    with raises(InvalidArgumentValueCombination):
        Tune(None, end=None)

    with raises(InvalidArgumentValue):
        Tune(end=10, changeType="bad")

@raises(ImproperObjectAction)
def test_Tune_immutable():
    crossVal = Tune([1, 2, 3])
    assert crossVal[1] == 2 # can get
    crossVal[1] = 0 # cannot set

def test_Tune_range_linear():
    lin = Tune(start=1, end=10, changeType='add')
    assert list(lin) == list(range(1, 11))

    lin = Tune(start=1, end=10, change=2)
    assert list(lin) == list(range(1, 11, 2))

    lin = Tune(start=10, end=1, change=-1)
    assert list(lin) == list(range(10, 0, -1))

    lin = Tune(start=1, end=10, change=9/49)
    assert np.allclose(lin, np.linspace(1, 10, 50))
    assert list(lin)[-1] == 10

    with raises(InvalidArgumentValue):
        Tune(start=1, end=10, change=-2)

    with raises(InvalidArgumentValue):
        Tune(start=10, end=1, change=2)

    with raises(InvalidArgumentValue):
        Tune(start=1, end=10, change=0)

def test_Tune_range_exponential():
    mul = Tune(start=0.0001, end=10, change=10, changeType='multiply')
    assert list(mul) == [0.0001, 0.001, 0.01, 0.1, 1, 10]

    mul = Tune(start=10, end=0.0001, change=0.1, changeType='multiply')
    assert np.allclose(mul, [10, 1, 0.1, 0.01, 0.001, 0.0001])

    with raises(InvalidArgumentValue):
        mul = Tune(start=0.0001, end=10, change=0.1, changeType='multiply')

    with raises(InvalidArgumentValue):
        mul = Tune(start=10, end=0.0001, change=10, changeType='multiply')




@noLogEntryExpected
def test_Tuning():
    default = Tuning()
    assert default.selection == "consecutive"
    assert default.validation == "cross validation"
    assert default._selector == Consecutive
    assert default._selectorArgs == {'loops': 1, 'order': None}
    assert default._validator == KFold
    assert default._validatorArgs == {'folds': 5}

    bfkf = Tuning(selection="brute force", folds=10)
    assert bfkf.selection == "brute force"
    assert bfkf.validation == "cross validation"
    assert bfkf._selector == BruteForce
    assert not bfkf._selectorArgs
    assert bfkf._validator == KFold
    assert bfkf._validatorArgs == {'folds': 10}

    conLoo = Tuning(validation="LeaveOneOut", loops=2, order=['a', 'b'])
    assert conLoo.selection == "consecutive"
    assert conLoo.validation == "LeaveOneOut"
    assert conLoo._selector == Consecutive
    assert conLoo._selectorArgs == {'loops': 2, 'order': ['a', 'b']}
    assert conLoo._validator == LeaveOneOut
    assert not conLoo._validatorArgs

    bfLogo = Tuning(selection="bruteforce", validation="leave one group out",
                    foldFeature="groups")
    assert bfLogo.selection == "bruteforce"
    assert bfLogo.validation == "leave one group out"
    assert bfLogo._selector == BruteForce
    assert not bfLogo._selectorArgs
    assert bfLogo._validator == LeaveOneGroupOut
    assert bfLogo._validatorArgs == {'foldFeature': "groups"}

    bfHod = Tuning(selection="bruteforce", validation="data",
                   validateX="ValX", validateY="ValY")
    # Note: validateX/Y won't work, but those are validated later by the
    # validator, not during Tuning init
    assert bfHod.selection == "bruteforce"
    assert bfHod.validation == "data"
    assert bfHod._selector == BruteForce
    assert not bfHod._selectorArgs
    assert bfHod._validator == HoldoutData
    assert bfHod._validatorArgs == {'validateX': "ValX", 'validateY': "ValY"}

    conProp = Tuning(validation=0.33)
    assert conProp.selection == "consecutive"
    assert conProp.validation == 0.33
    assert conProp._selector == Consecutive
    assert conProp._selectorArgs == {'loops': 1, 'order': None}
    assert conProp._validator == HoldoutProportion
    assert conProp._validatorArgs == {'proportion': 0.33}

    bayHod = Tuning(selection="bayesian", validation="data", validateX="ValX",
                    validateY="ValY", timeout=2400, threshold=0.000000001)
    assert bayHod.selection == "bayesian"
    assert bayHod.validation == "data"
    assert bayHod._selector == Bayesian
    assert bayHod._selectorArgs == {'maxIterations': 100, 'timeout': 2400,
                                    'threshold': 0.000000001}
    assert bayHod._validator == HoldoutData
    assert bfHod._validatorArgs == {'validateX': "ValX", 'validateY': "ValY"}

    itrHod = Tuning(selection="iterative", validation="data", validateX="ValX",
                    validateY="ValY", maxIterations=None, timeout=2400)
    assert itrHod.selection == "iterative"
    assert itrHod.validation == "data"
    assert itrHod._selector == Iterative
    assert itrHod._selectorArgs == {'maxIterations': None, 'timeout': 2400,
                                    'threshold': None}
    assert itrHod._validator == HoldoutData
    assert itrHod._validatorArgs == {'allowIncremental': True,
                                     'validateX': "ValX", 'validateY': "ValY"}

    def build():
        pass

    srmHod = Tuning(selection="storm", validation="data", validateX="ValX",
                    validateY="ValY", maxIterations=None, threshold=3,
                    timeout=None, learnerArgsFunc=build)
    assert srmHod.selection == "storm"
    assert srmHod.validation == "data"
    assert srmHod._selector == StochasticRandomMutator
    assert srmHod._selectorArgs == {'maxIterations': None, 'timeout': None,
                                    'threshold': 3,  'learnerArgsFunc': build,
                                    'initRandom': 5,
                                    'randomizeAxisFactor': 0.75}
    assert srmHod._validator == HoldoutData
    assert srmHod._validatorArgs == {'validateX': "ValX", 'validateY': "ValY"}

    with raises(InvalidArgumentValue):
        invalid = Tuning(selection="not a selection")

    with raises(InvalidArgumentValue):
        invalid = Tuning(validation="not a validation")

    with raises(InvalidArgumentValue):
        invalid = Tuning(validation=100)

    for selection in ['bayesian', 'iterative', 'storm']:
        with raises(InvalidArgumentValueCombination):
            invalid = Tuning(selection, .2) # require "data" validation

        with raises(InvalidArgumentValueCombination):
             # maxIterations, timeout, and threshold cannot all be None
            invalid = Tuning(selection, "data", validateX="ValX",
                             validateY="ValY", maxIterations=None)

def test_Tuning_performanceFunction():
    at = Tuning()
    X = nimble.ones(100, 11)
    Y = 0
    tuned = at.tune("nimble.KNNClassifier", X, Y, {}, fractionIncorrect, None,
                    False)
    assert at.performanceFunction == fractionIncorrect

    at = Tuning(performanceFunction=meanAbsoluteError)
    tuned = at.tune("nimble.KNNClassifier", X, Y, {}, fractionIncorrect, None,
                    False)
    assert at.performanceFunction == meanAbsoluteError

    with raises(InvalidArgumentValue):
        at = Tuning()
        tuned = at.tune("nimble.KNNClassifier", X, Y, {}, None, None, False)

@noLogEntryExpected
def test_Tuning_tune():
    X = nimble.random.data(100, 9, 0, useLog=False)
    Y = nimble.random.data(100, 1, 0, elementType='int', useLog=False)
    default = Tuning()
    default.tune("nimble.KNNClassifier", X, Y, {}, fractionIncorrect, None,
                 False)
    assert default._selector is Consecutive
    assert default._validator is KFold
    assert isinstance(default.validator, KFold)

    bfkf = Tuning(selection="brute force", folds=10)
    bfkf.tune("nimble.KNNClassifier", X, Y, {}, fractionIncorrect, None, False)
    assert bfkf._selector is BruteForce
    assert bfkf._validator is KFold
    assert isinstance(bfkf.validator, KFold)

    conLoo = Tuning(validation="leave-one-out", loops=2)
    conLoo.tune("nimble.KNNClassifier", X, Y, {}, fractionIncorrect, None,
                False)
    assert conLoo._selector is Consecutive
    assert conLoo._validator is LeaveOneOut
    assert isinstance(conLoo.validator, LeaveOneOut)

    bfLogo = Tuning(selection="bruteforce", validation="leaveonegroupout",
                    foldFeature="groups")
    X.features.setName(2, "groups", useLog=False)
    bfLogo.tune("nimble.KNNClassifier", X, Y, {}, fractionIncorrect, None,
                False)
    assert bfLogo._selector is BruteForce
    assert bfLogo._validator is LeaveOneGroupOut
    assert isinstance(bfLogo.validator, LeaveOneGroupOut)

    bfHod = Tuning(selection="bruteforce", validation="data", validateX=X,
                   validateY=Y)
    bfHod.tune("nimble.KNNClassifier", X, Y, {}, fractionIncorrect,
                        None, False)
    assert bfHod._selector is BruteForce
    assert bfHod._validator is HoldoutData
    assert isinstance(bfHod.validator, HoldoutData)

    conProp = Tuning(validation=0.33)
    conProp.tune("nimble.KNNClassifier", X, Y, {}, fractionIncorrect, None,
                 False)
    assert conProp._selector is Consecutive
    assert conProp._validator is HoldoutProportion
    assert isinstance(conProp.validator, HoldoutProportion)
