
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
Functions (and their helpers) used to analyze arbitrary performance
functions.
"""

import math
import functools

import numpy as np

import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.random import numpyRandom

ACCEPTED_STATS = [
            'max', 'mean', 'median', 'min', 'unique count',
            'proportion missing', 'proportion zero', 'standard deviation',
            'std', 'population std', 'population standard deviation',
            'sample std', 'sample standard deviation'
            ]

def detectBestResult(functionToCheck):
    """
    Determine if higher or lower values are optimal for the function.

    Provides sample data to the ``functionToCheck`` and evaluates the
    results to determine whether the function associates correctness of
    predictions with minimum returned values or maximum returned values.
    The trials are avoided if ``functionToCheck`` has an 'optimal'
    attribute with the value 'min' or 'max'. The optimal values are
    'min' if lower values are associated with correctness of predictions
    by ``functionToCheck``, or 'max' if higher values are associated
    with correctness.

    Functions are expected to take two parameters, knownValues and
    predictedValues. The knownValues are the known data which will be
    compared to the predictedValues. The predictedValues should accept
    data in one of three forms, depending whether the 'scoreMode' is set
    to 'labels', 'bestScore', or 'allScores'. This function will test
    also determine the 'scoreMode' expected by ``functionToCheck``.

    Parameters
    ----------
    functionToCheck : function
        In the form function(knownValues, predictedValues) that is
        expected to return a single numeric value.
    """
    if hasattr(functionToCheck, 'optimal'):
        if functionToCheck.optimal == 'min':
            return 'min'
        if functionToCheck.optimal == 'max':
            return 'max'

    def validResult(toCheck):
        if toCheck == 'nan':
            return False
        if isinstance(toCheck, Exception):
            return False
        return True

    resultsByType = [None, None, None]
    trialSize = 10
    # we can't know which format is expected in the predicted values param,
    # so we try all three and then interpret the results
    # (0) if predicted labels, (1) if bestScores, (2) if allScores
    for predictionType in [0, 1, 2]:
        # Run the main trial, which uses a mixture of ones and zeros
        knowns = _generateMixedRandom(trialSize)
        try:
            result = _runTrialGivenParameters(functionToCheck, knowns,
                                              predictionType)
        except Exception as e: # pylint: disable=broad-except
            result = e

        # If the trial ends successfully, then we do further trials for
        # consistency
        if validResult(result):
            # for bestScores and allscores, there is a lot more random
            # generation, so we run a bundle of trials for consistency. For
            # labels we only run one extra trial.
            confidenceTrials = 1 if predictionType == 0 else 10

            # Since we are doing randomly generated data, if the performance
            # function is only considering subsets of the data, then it is
            # possible for us to generate numbers cause weirdness or outright
            # failures. We allow for one such result
            freebieAvailable = not predictionType
            for _ in range(confidenceTrials):
                knownsMixed = _generateMixedRandom(trialSize)
                try:
                    resultMixed = _runTrialGivenParameters(functionToCheck,
                                                           knownsMixed,
                                                           predictionType)
                except Exception as e: # pylint: disable=broad-except
                    resultMixed = e

                if resultMixed != result:
                    if freebieAvailable:
                        freebieAvailable = False
                    else:
                        # inconsistent but valid results
                        if validResult(resultMixed):
                            msg = "In repeated trials with different known "
                            msg += "values, functionToCheck was inconsistent "
                            msg += "in attributing high versus low values to "
                            msg += "optimal predictions."
                            result = InvalidArgumentValue(msg)
                        # erroneous call on what should be acceptable data;
                        # report the error to the user
                        else:
                            result = resultMixed
                            break

            # in labels case we additionally check against some simpler data
            if predictionType == 0:
                knownsZeros = _generateAllZeros(trialSize)
                try:
                    resultZeros = _runTrialGivenParameters(functionToCheck,
                                                           knownsZeros,
                                                           predictionType)
                except Exception as e: # pylint: disable=broad-except
                    resultZeros = e
                knownsOnes = _generateAllOnes(trialSize)
                try:
                    resultOnes = _runTrialGivenParameters(functionToCheck,
                                                          knownsOnes,
                                                          predictionType)
                except Exception as e: # pylint: disable=broad-except
                    resultOnes = e

                # check knownsZeros results same as knowns ones results
                rzValid = (resultZeros != 'nan'
                           and not isinstance(resultZeros, Exception))
                roValid = (resultOnes != 'nan'
                           and not isinstance(resultOnes, Exception))
                if rzValid and roValid:
                    if resultZeros != resultOnes:
                        msg = "Over trials with different knowns, "
                        msg += "functionToCheck was inconsistent in "
                        msg += "attributing high versus low values to optimal "
                        msg += "predictions."
                        result = InvalidArgumentValue(msg)

        # record the result, regardless of whether it was successful, nan,
        # or an exception
        resultsByType[predictionType] = result

    best = None
    for result in resultsByType:
        if result != 'nan' and not isinstance(result, Exception):
            # inconsistent results
            if best is not None and result != best:
                reason = "Trials were run with all possible formats for "
                reason += "predicted values, but gave inconsistent results. "
                detectBestException(reason, resultsByType)
            best = result

    # No valid results / all trials resulted in exceptions
    if best is None:
        reason = "Trials were run with all possible formats for predicted "
        reason += "values, but none gave valid results. "
        detectBestException(reason, resultsByType)

    return best


def detectBestException(preface, outputs):
    """
    Exception generator when detectBestResult is unsuccessful.
    """
    msg = "Either functionToCheck has bugs, is not a performance function, "
    msg += "or is incompatible with these trials. These trials can be avoided "
    msg += "if the user manually declares which kinds of values are "
    msg += "associated with correct predictions (either 'min' or 'max') "
    msg += "by adding an attribute named 'optimal' to functionToCheck. For "
    msg += "debugging purposes, the trial results per format are given:"
    msg += " (labels) " + str(outputs[0])
    msg += " (best scores) " + str(outputs[1])
    msg += " (all scores) " + str(outputs[2])

    raise InvalidArgumentValue(preface + msg)


def _runTrialGivenParameters(toCheck, knowns, predictionType):
    predicted = _generatePredicted(knowns, predictionType)

    allCorrectScore = toCheck(knowns.copy(), predicted)
    # this is going to hold the output of the function. Each value will
    # correspond to a trial that contains incrementally less correct predicted
    # values
    scoreList = [allCorrectScore]
    # range over the indices of predicted values, making them incorrect one
    # by one
    for index in range(len(predicted.points)):
        _makeIncorrect(predicted, predictionType, index)
        scoreList.append(toCheck(knowns.copy(), predicted))

    allWrongScore = scoreList[len(scoreList) - 1]

    for score in scoreList:
        if math.isnan(score):
            return "nan"

    if allWrongScore < allCorrectScore:
        prevScore = allCorrectScore + 1
        for score in scoreList:
            if (score < allWrongScore
                    or score > allCorrectScore
                    or score > prevScore):
                msg = "all-correct and all-incorrect trials indicated max "
                msg += "optimality, but mixed correct/incorrect scores were "
                msg += "not monotonicly increasing with correctness"
                raise _NonMonotonicResultsException(msg)
            prevScore = score
        return "max"
    if allWrongScore > allCorrectScore:
        prevScore = allCorrectScore - 1
        for score in scoreList:
            if (score > allWrongScore
                    or score < allCorrectScore
                    or score < prevScore):
                msg = "all-correct and all-incorrect trials indicated min "
                msg += "optimality, but mixed correct/incorrect scores were "
                msg += "not monotonicly decreasing with correctness"
                raise _NonMonotonicResultsException(msg)
            prevScore = score
        return "min"
    # allWrong and allCorrect must not be equal, otherwise it cannot be a
    # measure of correct performance
    msg = "functionToCheck produced the same values for trials "
    msg += "including all-correct and all-incorrect predictions."
    raise _NoDifferenceResultsException(msg)


def _generateAllZeros(length):
    return nimble.data(np.zeros([length, 1], dtype=int),
                       useLog=False)


def _generateAllOnes(length):
    return nimble.data(np.ones([length, 1], dtype=int),
                       useLog=False)


def _generateMixedRandom(length):
    while True:
        correct = numpyRandom.randint(2, size=[length, 1])
        # we don't want all zeros or all ones
        if np.any(correct) and not np.all(correct):
            break
    correct = nimble.data(source=correct, useLog=False)
    return correct


def _generatePredicted(knowns, predictionType):
    """
    Predicted may mean any of the three kinds of output formats for
    trainAndApply: predicted labels, bestScores, or allScores.
    If confidences are involved, they are randomly generated, yet
    consistent with correctness
    """
    workingCopy = knowns.copy()
    workingCopy.features.setNames('PredictedClassLabel', oldIdentifiers=0, useLog=False)
    # Labels
    if predictionType == 0:
        return workingCopy
    # Labels and the score for that label (aka 'bestScores')
    if predictionType == 1:
        scores = numpyRandom.randint(2, size=[len(workingCopy.points), 1])
        scores = nimble.data(source=scores,
                             featureNames=['LabelScore'], useLog=False)
        workingCopy.features.append(scores, useLog=False)
        return workingCopy
    # Labels, and scores for all possible labels (aka 'allScores')
    dataToFill = []
    for i in range(len(workingCopy.points)):
        currConfidences = [None, None]
        winner = numpyRandom.randint(10) + 10 + 2
        loser = numpyRandom.randint(winner - 2) + 2
        if knowns._data[i][0] == 0:
            currConfidences[0] = winner
            currConfidences[1] = loser
        else:
            currConfidences[0] = loser
            currConfidences[1] = winner
        dataToFill.append(currConfidences)

    scores = nimble.data(source=dataToFill,
                         featureNames=['0', '1'], useLog=False)
    return scores


def _makeIncorrect(predicted, predictionType, index):
    if predictionType in [0, 1]:
        predicted._data[index][0] = math.fabs(predicted._data[index][0] - 1)
    else:
        temp = predicted._data[index][0]
        predicted._data[index][0] = predicted._data[index][1]
        predicted._data[index][1] = temp


class _NonMonotonicResultsException(Exception):
    """
    Exception to be thrown if the results returned by a performance
    function do not trend monotoniclly up or down as predicted
    data approach known values.
    """

    def __init__(self, value):
        self.value = value
        super().__init__()

    def __str__(self):
        return repr(self.value)


class _NoDifferenceResultsException(Exception):
    """
    Exception to be thrown if the result returned by a performance
    function when given all-incorrect predicted data exactly match
    the result when the performance function is given all-correct
    prediction data.
    """

    def __init__(self, value):
        self.value = value
        super().__init__()

    def __str__(self):
        return repr(self.value)


def performanceFunction(optimal, best=None, predict=None, validate=True,
                        requires1D=True, samePtCount=True, sameFtCount=True,
                        allowEmpty=False, allowMissing=False):
    """
    Decorator factory for Nimble performance functions.

    A convenient way to make a function compatible with Nimble's testing
    API. The function that will be decorated must take the form:
    ``function(knownValues, predictedValues)`` as these inputs are
    provided by Nimble during testing (including validation testing).
    The ``optimal``, ``best``, and ``predict`` become attributes of the
    function that allow Nimble to compare and analyze the performances.
    The remaining parameters control validations of the data input to
    the decorated function. By default, both knownValues and
    predictedValues must be non-empty Nimble data objects with one
    feature, an equal number of points, and no missing values.

    The ``predict`` parameter informs Nimble how to generate the
    predictedValues for the decorated function. Values of None,
    'bestScore' and 'allScores' utilize ``TrainedLearner.apply`` with
    ``predict`` providing the scoreMode argument. More complex cases
    are handled by providing a custom function to ``predict`` which must
    be in the form: ``predict(trainedLearner, knownValues, arguments)``.
    With access to the TrainedLearner instance, knownValues, and the
    arguments provided to the testing function it should be possible to
    generate the desired predicted values.

    Note
    ----
    Common performance functions are available in nimble.calculate.

    Parameters
    ----------
    optimal : str
        Either 'max' or 'min' indicating whether higher or lower values
        are better.
    best : int, float, None
        The best possible value for the performance function. None
        assumes that the values have no bound in the optimal direction.
    predict : str, function
        Informs Nimble how to produce the predictedValues. May be None,
        'bestScore', or 'allScores' to utilize TrainedLearner.apply() or
        a custom function in the form:
        predict(trainedLearner, knownValues, arguments)
    validate : bool
        Whether to perform validation on the function inputs. If False,
        none of the parameters below will apply.
    requires1D : bool
        Checks that the predictedValues object is one-dimensional.
    samePtCount : bool
        Checks that the knownValues and predictedValues have the same
        number of points.
    sameFtCount : bool
        Checks that the knownValues and predictedValues have the same
        number of features.
    allowEmpty : bool
        Allow the knownValues and predictedValues objects to be empty.
    allowMissing : bool
        Allow the knownValues and predictedValues to contain missing
        values.

    See Also
    --------
    nimble.calculate

    Examples
    --------
    Here, ``correctVoteRatio`` finds the number of times the correct
    label received a vote and divides by the total number of votes. The
    best score is 1, indicating that 100% of the votes were for the
    correct labels. As inputs it expects the knownValues to be a feature
    vector of known labels and the predictedValues to be a matrix of
    vote counts for each label.

    >>> @performanceFunction('max', 1, 'allScores', requires1D=False,
    ...                      sameFtCount=False)
    ... def correctVoteRatio(knownValues, predictedValues):
    ...     cumulative = 0
    ...     totalVotes = 0
    ...     for true, votes in zip(knownValues, predictedValues.points):
    ...         cumulative += votes[true]
    ...         totalVotes += sum(votes)
    ...     return cumulative / totalVotes
    ...
    >>> trainX = nimble.data([[0, 0], [2, 2], [-2, -2]] * 10)
    >>> trainY = nimble.data([0, 1, 2] * 10).T
    >>> testX = nimble.data([[0, 0], [1, 1], [2, 2], [1, 1], [-1, -2]])
    >>> testY = nimble.data([0, 0, 1, 1, 2]).T
    >>> knn = nimble.train('nimble.KNNClassifier', trainX, trainY, k=3)
    >>> # Can visualize our predictedValues for this case using apply()
    >>> knn.apply(testX, scoreMode='allScores') # 12/15 votes correct
    <Matrix 5pt x 3ft
         0  1  2
       ┌────────
     0 │ 3  0  0
     1 │ 2  1  0
     2 │ 0  3  0
     3 │ 2  1  0
     4 │ 0  0  3
    >
    >>> knn.test(correctVoteRatio, testX, testY)
    0.8

    Here, ``averageDistanceToCenter`` calculates the average distance of
    a point to its center. This function expects the predictedValues to
    be the cluster center for the predicted label of each point in the
    knownValues. The string options for 'predict' do not cover this
    case, so it requires the ``labelsWithCenters`` function.

    >>> def labelsWithCenters(trainedLearner, knownValues, arguments):
    ...     labels = trainedLearner.apply(knownValues)
    ...     centers = trainedLearner.getAttributes()['cluster_centers_']
    ...     return nimble.data([centers[l] for l in labels])
    ...
    >>> @performanceFunction('min', 0, predict=labelsWithCenters,
    ...                      requires1D=False, sameFtCount=False)
    ... def averageDistanceToCenter(knownValues, predictedValues):
    ...     rootSqDiffs = ((knownValues - predictedValues) ** 2) ** 0.5
    ...     distances = rootSqDiffs.points.calculate(lambda pt: sum(pt))
    ...     return sum(distances) / len(distances.points)
    ...
    >>> X = nimble.data([[0, 0], [4, 0], [0, 4], [4, 4]] * 25)
    >>> X += nimble.random.data(100, 2, 0, randomSeed=1) # add noise
    >>> round(nimble.trainAndTest('skl.KMeans', averageDistanceToCenter, X,
    ...                     n_clusters=4, randomSeed=1), 6)
    1.349399
    """
    if optimal not in ['min', 'max']:
        raise InvalidArgumentValue('optimal must be "min" or "max"')
    if not callable(predict):
        if predict is None or predict.lower() in ['bestscore', 'allscores']:
            scoreMode = predict
            # pylint: disable=function-redefined
            def predict(trainedLearner, knownValues, arguments):
                return trainedLearner.apply(knownValues, arguments, scoreMode,
                                            useLog=False)
        else:
            msg = 'predict must be callable or a valid scoreMode (None, '
            msg += '"bestScore", "allScores") for TrainedLearner.apply '
            raise InvalidArgumentType(msg)

    def performanceFunctionDecorator(func):
        func.optimal = optimal
        func.best = best
        func.predict = predict
        if not validate:
            return func

        @functools.wraps(func)
        def wrapped(knownValues, predictedValues):
            if not isinstance(knownValues, nimble.core.data.Base):
                msg = "knownValues is not a Nimble data object"
                raise InvalidArgumentType(msg)
            if not isinstance(predictedValues, nimble.core.data.Base):
                msg = "predictedValues is not a Nimble data object"
                raise InvalidArgumentType(msg)
            if (not allowEmpty and (0 in knownValues.shape
                                    or 0 in predictedValues.shape)):
                msg = 'Cannot calculate performance on an empty object'
                raise InvalidArgumentValue(msg)
            if (samePtCount and (len(knownValues.points)
                                 != len(predictedValues.points))):
                msg = "The known and predicted data must have the same number "
                msg += "of points"
                raise InvalidArgumentValueCombination(msg)
            if (sameFtCount and (len(knownValues.features)
                                 != len(predictedValues.features))):
                msg = "The known and predicted data must have the same number "
                msg += "of features"
                raise InvalidArgumentValueCombination(msg)
            if requires1D and len(predictedValues.features) > 1:
                msg = "predictedValues must be labels only; this has more "
                msg += "than one feature"
                raise InvalidArgumentValue(msg)
            if not allowMissing and match.anyMissing(knownValues):
                msg = "Unable to calculate the performance because the  "
                msg += "knownValues contain missing data."
                raise InvalidArgumentValue(msg)
            if not allowMissing and match.anyMissing(predictedValues):
                msg = "Unable to calculate the performance because the  "
                msg += "predictedValues contain missing data."
                raise InvalidArgumentValue(msg)

            return func(knownValues, predictedValues)

        return wrapped

    return performanceFunctionDecorator
