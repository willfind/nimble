
import nimble
from nimble import Tune, Tuning
from nimble.core.tune import KFold, LeaveOneOut, LeaveOneGroupOut
from nimble.core.tune import HoldoutData, HoldoutProportion
from nimble.core.tune import BruteForce, Consecutive # Bayesian, Iterative
from nimble.calculate import fractionIncorrect, meanAbsoluteError
from nimble.exceptions import InvalidArgumentValue, ImproperObjectAction

from tests.helpers import raises, noLogEntryExpected

def test_Tune():
    crossVal = Tune([1, 2, 3])
    assert len(crossVal) == 3
    assert crossVal[1] == 2
    assert str(crossVal) == "(1, 2, 3)"
    assert repr(crossVal) == "Tune([1, 2, 3])"

@raises(ImproperObjectAction)
def test_Tune_immutable():
    crossVal = Tune([1, 2, 3])
    assert crossVal[1] == 2 # can get
    crossVal[1] = 0 # cannot set

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
                   validateX="TrainX", validateY="TrainY")
    # Note: validateX/Y won't work, but those are validated later by the
    # validator, not during Tuning init
    assert bfHod.selection == "bruteforce"
    assert bfHod.validation == "data"
    assert bfHod._selector == BruteForce
    assert not bfHod._selectorArgs
    assert bfHod._validator == HoldoutData
    assert bfHod._validatorArgs == {'validateX': "TrainX",
                                     'validateY': "TrainY"}

    conProp = Tuning(validation=0.33)
    assert conProp.selection == "consecutive"
    assert conProp.validation == 0.33
    assert conProp._selector == Consecutive
    assert conProp._selectorArgs == {'loops': 1, 'order': None}
    assert conProp._validator == HoldoutProportion
    assert conProp._validatorArgs == {'proportion': 0.33}

    with raises(InvalidArgumentValue):
        invalid = Tuning(selection="not a selection")

    with raises(InvalidArgumentValue):
        invalid = Tuning(validation="not a validation")

    with raises(InvalidArgumentValue):
        invalid = Tuning(validation=100)

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
