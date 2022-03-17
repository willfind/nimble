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
from nimble.calculate import performanceFunction
from nimble.calculate.loss import _computeError
from nimble.exceptions import InvalidArgumentValue, InvalidArgumentType
from nimble.exceptions import InvalidArgumentValueCombination
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


#######################
# performanceFunction #
#######################

@performanceFunction('min')
def genericValidated(knownValues, predictedValues):
    return 1

def testGenericPerformanceFunctionEmpty():
    """
    Test that performanceFunction raises an exception if data is empty
    """
    emptyArray = np.array([])
    nonEmptyArray = np.array([1, 2, 3])
    emptyMatrix = nimble.data(source=emptyArray)
    nonEmptyMatrix = nimble.data(source=nonEmptyArray)

    with raises(InvalidArgumentValue):
        perf = genericValidated(emptyMatrix, emptyMatrix)

    with raises(InvalidArgumentValue):
        perf = genericValidated(nonEmptyMatrix, emptyMatrix)

    with raises(InvalidArgumentValue):
        perf = genericValidated(emptyMatrix, nonEmptyMatrix)

@raises(InvalidArgumentValue)
def testGenericPerformanceFunctionNansNoData():
    """
    Test that performanceFunction raises an exception if nans prevent calculation.
    """
    knownLabels = np.array([1, np.nan, 3])
    predictedLabels = np.array([np.nan, 3, np.nan])

    knownLabelsMatrix = nimble.data(source=knownLabels)
    predictedLabelsMatrix = nimble.data(source=predictedLabels)

    perf = genericValidated(knownLabelsMatrix, predictedLabelsMatrix)

def testGenericPerformanceFunctionInvalidInputs():
    """
    Test that performanceFunction raises an exception for invalid types
    """
    with raises(InvalidArgumentType):
        knownLabels = np.array([[1], [2], [3]])
        predictedLabels = np.array([[1], [2], [3]])

        predictedLabelsMatrix = nimble.data(source=predictedLabels)

        perf = genericValidated(knownLabels, predictedLabelsMatrix)

    with raises(InvalidArgumentType):
        knownLabels = np.array([[1], [2], [3]])
        predictedLabels = np.array([[1], [2], [3]])

        knownLabelsMatrix = nimble.data(source=knownLabels)

        perf = genericValidated(knownLabelsMatrix, predictedLabels)

def testGenericPerformanceFunctionShapeMismatch():
    """
    Test that performanceFunction raises an exception if shapes don't match
    """
    with raises(InvalidArgumentValueCombination):
        knownLabels = np.array([[1], [2], [3]])
        predictedLabels = np.array([[1], [2]])

        knownLabelsMatrix = nimble.data(source=knownLabels)
        predictedLabelsMatrix = nimble.data(source=predictedLabels)

        perf = genericValidated(knownLabelsMatrix, predictedLabelsMatrix)

    with raises(InvalidArgumentValueCombination):
        knownLabels = np.array([[1], [2]])
        predictedLabels = np.array([[1], [2], [3]])

        knownLabelsMatrix = nimble.data(source=knownLabels)
        predictedLabelsMatrix = nimble.data(source=predictedLabels)

        perf = genericValidated(knownLabelsMatrix, predictedLabelsMatrix)

    with raises(InvalidArgumentValueCombination):
        knownLabels = np.array([[1, 1], [2, 2], [3, 3]])
        predictedLabels = np.array([[1], [2], [3]])

        knownLabelsMatrix = nimble.data(source=knownLabels)
        predictedLabelsMatrix = nimble.data(source=predictedLabels)

        perf = genericValidated(knownLabelsMatrix, predictedLabelsMatrix)

    with raises(InvalidArgumentValueCombination):
        knownLabels = np.array([[1], [2], [3]])
        predictedLabels = np.array([[1, 1], [2, 2], [3, 3]])

        knownLabelsMatrix = nimble.data(source=knownLabels)
        predictedLabelsMatrix = nimble.data(source=predictedLabels)

        perf = genericValidated(knownLabelsMatrix, predictedLabelsMatrix)


def testGenericPerformanceFunction2D():
    """
    Test that performanceFunction checks allowed input shape
    """

    with raises(InvalidArgumentValue):

        knownLabels = np.array([[1, 2], [2, 3], [3, 4]])
        predictedLabels = np.array([[1, 2], [2, 3], [3, 4]])

        knownLabelsMatrix = nimble.data(source=knownLabels)
        predictedLabelsMatrix = nimble.data(source=predictedLabels)

        perf = genericValidated(knownLabelsMatrix, predictedLabelsMatrix)

    @performanceFunction('min', requires1D=False)
    def generic2D(knownValues, predictedValues):
        return 1

    knownLabels = np.array([[1, 2], [2, 3], [3, 4]])
    predictedLabels = np.array([[1, 2], [2, 3], [3, 4]])

    knownLabelsMatrix = nimble.data(source=knownLabels)
    predictedLabelsMatrix = nimble.data(source=predictedLabels)

    perf = generic2D(knownLabelsMatrix, predictedLabelsMatrix)
    assert perf == 1

@noLogEntryExpected
def testGenericPerformanceFunctionWithValidData():
    """
    Test that performanceFunction wrapper works as expected
    """
    knownLabels = [[1.0], [3.0], [0.0], [4.0]]
    predictedLabels = [[1.0], [2.0], [0.0], [3.0]]

    knownLabelsMatrix = nimble.data(source=knownLabels, useLog=False)
    predictedLabelsMatrix = nimble.data(source=predictedLabels, useLog=False)

    @performanceFunction('min', 0)
    def performance(knownValues, predictedValues):
        return _computeError(knownValues, predictedValues,
                             lambda x, y, z: z + (x - y), lambda x, y: x / y)

    assert performance.optimal == 'min'
    assert performance.best == 0
    perf = performance(knownLabelsMatrix, predictedLabelsMatrix)
    assert perf == 0.5

def testGenericPerformanceFunctionNoValidation():
    """
    Test that a performance function that does not require validation
    """
    knownLabels = np.array([[1, 2], [3, 4]])
    predictedLabels = np.array([[1], [2], [3]])

    knownLabelsMatrix = nimble.data(source=knownLabels)
    predictedLabelsMatrix = nimble.data(source=predictedLabels)

    @performanceFunction('max', 1, validate=False)
    def noValidation(knownValues, predictedValues):
        return 1

    assert noValidation.optimal == 'max'
    assert noValidation.best == 1
    perf = noValidation(knownLabelsMatrix, predictedLabelsMatrix)
    assert perf == 1
