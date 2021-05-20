import numpy as np

import nimble
from nimble.calculate import cosineSimilarity
from nimble.calculate import detectBestResult
from nimble.calculate import fractionCorrect
from nimble.calculate import fractionIncorrect
from nimble.calculate import meanAbsoluteError
from nimble.calculate import rootMeanSquareError
from nimble.calculate import rSquared
from nimble.calculate import varianceFractionRemaining
from nimble.exceptions import InvalidArgumentValue
from tests.helpers import raises
from tests.helpers import noLogEntryExpected

####################
# detectBestResult #
####################

# labels, best, all
# inconsistent between different mixed runs

@raises(InvalidArgumentValue)
def test_detectBestResult_labels_inconsistentForDifferentKnowns():
    def foo(knowns, predicted):
        rawKnowns = knowns.copy(to="numpyarray")
        rawPred = predicted.copy(to="np.array")

        # case: knowns all zeros
        if not np.any(rawKnowns):
            # case: predictd all right
            if not np.any(rawPred):
                return 0
            # case: predicted all wrong
            elif np.all(rawPred):
                return 1
            else:
                return nimble.calculate.meanAbsoluteError(knowns, predicted)
        # case: knowns all ones
        elif np.all(rawKnowns):
            # case: predicted all wrong
            if not np.any(rawPred):
                return 0
            # case: predictd all right
            elif np.all(rawPred):
                return 1
            else:
                return 1 - nimble.calculate.meanAbsoluteError(knowns, predicted)

        return nimble.calculate.meanAbsoluteError(knowns, predicted)

    detectBestResult(foo)


@raises(InvalidArgumentValue)
def test_detectBestResult_labels_allcorrect_equals_allwrong():
    detectBestResult(lambda x, y: 20)


@raises(InvalidArgumentValue)
def test_detectBestResult_labels_nonmonotonic_minmizer():
    def foo(knowns, predicted):
        ret = nimble.calculate.fractionIncorrect(knowns, predicted)
        if ret > .25 and ret < .5:
            return .6
        if ret >= .5 and ret < .75:
            return .4
        return ret

    detectBestResult(foo)


@raises(InvalidArgumentValue)
def test_detectBestResult_labels_nonmonotonic_maxizer():
    def foo(knowns, predicted):
        if knowns == predicted:
            return 100
        else:
            return nimble.random.pythonRandom.randint(0, 20)

    detectBestResult(foo)


@raises(InvalidArgumentValue)
def test_detectBestResult_wrongSignature_low():
    def tooFew(arg1):
        return 0

    detectBestResult(tooFew)


@raises(InvalidArgumentValue)
def test_detectBestResult_wrongSignature_high():
    def tooMany(arg1, arg2, arg3):
        return 0

    detectBestResult(tooMany)


def test_detectBestResult_exceptionsAreReported():
    wanted = "SPECIAL TEXT"

    def neverWorks(knowns, predicted):
        raise InvalidArgumentValue(wanted)

    with raises(InvalidArgumentValue, match=wanted):
        detectBestResult(neverWorks)

@noLogEntryExpected
def _backend(performanceFunction, optimality):
    assert performanceFunction.optimal == optimality
    try:
        performanceFunction.optimal = None
        result = detectBestResult(performanceFunction)
        assert result == optimality
    finally:
        performanceFunction.optimal = optimality


def test_detectBestResult_rootMeanSquareError():
    _backend(rootMeanSquareError, 'min')


def test_detectBestResult_meanAbsoluteError():
    _backend(meanAbsoluteError, 'min')


def test_detectBestResult_fractionCorrect():
    _backend(fractionCorrect, 'max')


def test_detectBestResult_fractionIncorrect():
    _backend(fractionIncorrect, 'min')


def test_detectBestResult_cosineSimilarity():
    _backend(cosineSimilarity, 'max')


def test_detectBestResult_rSquared():
    _backend(rSquared, 'max')


def test_detectBestResult_varianceFractionRemaining():
    _backend(varianceFractionRemaining, 'min')
