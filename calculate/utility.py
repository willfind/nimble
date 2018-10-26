"""
Functions (and their helpers) used to analyze arbitrary performance
functions.

"""

from __future__ import absolute_import
import inspect
import numpy
import math

import UML
from UML.exceptions import ArgumentException
from UML.randomness import numpyRandom
from six.moves import range


def detectBestResult(functionToCheck):
    """
    Provides sample data to the function in question and evaluates the results
    to determine whether the function associates correctness of predictions with
    minimum returned values or maximum returned values. If the user wants these
    trials to be avoided, then they can add an attribute named 'optimal' to
    functionToCheck. Two values are accepted: 'min' if lower values are
    associated with correctness of predictions by functionToCheck, or 'max' if
    higher values are associated with correctness. If the attribute 'optimal'
    is set to any other value then the trials are run in the same way as if
    functionToCheck had no attribute by that name.

    functionToCheck may only take two or three arguments. In the two argument
    case, the first must be a vector of desired values and the second must be
    a vector of predicted values. In the second case, the first argument must
    be a vector of known labes, the second argument must be an object
    containing confidence scores for different labels, and the third argument
    must be the value of a label value present in the data. In either cause,
    the functions must return a float value.

    returns: Either 'min' or 'max'; the former if lower values returned from
    functionToCheck are associated with correctness of predictions, the later
    if larger values are associated with correctness. If we are unable to
    determine which is correct, then an ArgumentException is thrown.
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
            result = _runTrialGivenParameters(functionToCheck, knowns, predictionType)
        except Exception as e:
            result = e

        # If the trial ends successfully, then we do further trials for
        # consistency
        if validResult(result):
            # for bestScores and allscores, there is a lot more random generation,
            # so we run a bundle of trials for consistency. For labels we only
            # run one extra trial.
            confidenceTrials = 1 if predictionType == 0 else 10

            # Since we are doing randomly generated data, if the performance
            # funciton is only considering subsets of the data, then it is
            # possible for us to generate numbers cause weirdness or outright
            # failures. We allow for one such result
            freebieAvailable = False if predictionType == 0 else True
            for i in range(confidenceTrials):
                knownsMixed = _generateMixedRandom(trialSize)
                try:
                    resultMixed = _runTrialGivenParameters(functionToCheck, knownsMixed, predictionType)
                except Exception as e:
                    resultMixed = e

                if resultMixed != result:
                    if freebieAvailable:
                        freebieAvailable = False
                    else:
                        # inconsistent but valid results
                        if validResult(resultMixed):
                            msg = "In repeated trials with different known values, "
                            msg += "functionToCheck was inconsistent in attributing "
                            msg += "high versus low values to optimal predictions."
                            result = ArgumentException(msg)
                        # erroneous call on what should be acceptable data;
                        # report the error to the user
                        else:
                            result = resultMixed
                            break

            # in the labels case we additionally check against some simpler data
            if predictionType == 0:
                knownsZeros = _generateAllZeros(trialSize)
                try:
                    resultZeros = _runTrialGivenParameters(functionToCheck, knownsZeros, predictionType)
                except Exception as e:
                    resultZeros = e
                knownsOnes = _generateAllOnes(trialSize)
                try:
                    resultOnes = _runTrialGivenParameters(functionToCheck, knownsOnes, predictionType)
                except Exception as e:
                    resultOnes = e

                # check knownsZeros results same as knowns ones results
                rzValid = resultZeros != 'nan' and not isinstance(resultZeros, Exception)
                roValid = resultOnes != 'nan' and not isinstance(resultOnes, Exception)
                if rzValid and roValid:
                    if resultZeros != resultOnes:
                        msg = "Over trials with different knowns, "
                        msg += "functionToCheck was inconsistent in attributing "
                        msg += "high versus low values to optimal predictions."
                        result = ArgumentException(msg)

        # record the result, regardless of whether it was successful, nan,
        # or an exception
        resultsByType[predictionType] = result

    best = None
    for result in resultsByType:
        if result != 'nan' and not isinstance(result, Exception):
            # inconsistent results
            if best is not None and result != best:
                reason = "Trials were run with all possible formats for predicted "
                reason += "values, but gave inconsistent results. "
                raiseException(reason, resultsByType)
            best = result

    # No valid results / all trials resulted in exceptions
    if best is None:
        reason = "Trials were run with all possible formats for predicted "
        reason += "values, but none gave valid results. "
        raiseException(reason, resultsByType)

    return best


def raiseException(preface, outputs):
    msg = "Either functionToCheck has bugs, is not a performance function, "
    msg += "or is incompatible with these trials. These trials can be avoided "
    msg += "if the user manulally declares which kinds of values are associated "
    msg += "with correct predictions (either 'min' or 'max') "
    msg += "by adding an attribute named 'optimal' to functionToCheck. For "
    msg += "debugging purposes, the trial results per format are given:"
    msg += " (labels) " + str(outputs[0])
    msg += " (best scores) " + str(outputs[1])
    msg += " (all scores) " + str(outputs[2])

    raise ArgumentException(preface + msg)


def _runTrialGivenParameters(toCheck, knowns, predictionType):
    predicted = _generatePredicted(knowns, predictionType)

    allCorrectScore = toCheck(knowns.copy(), predicted)
    # this is going to hold the output of the function. Each value will
    # correspond to a trial that contains incrementally less correct predicted values
    scoreList = [allCorrectScore]
    # range over the indices of predicted values, making them incorrect one
    # by one
    for index in range(predicted.points):
        _makeIncorrect(predicted, predictionType, index)
        scoreList.append(toCheck(knowns.copy(), predicted))

    allWrongScore = scoreList[len(scoreList) - 1]

    for score in scoreList:
        if math.isnan(score):
            return "nan"

    if allWrongScore < allCorrectScore:
        prevScore = allCorrectScore + 1
        for score in scoreList:
            if score < allWrongScore or score > allCorrectScore or score > prevScore:
                msg = "all-correct and all-incorrect trials indicated max optimality, "
                msg += "but mixed correct/incorrect scores were not monotonicly "
                msg += "increasing with correctness"
                raise _NonMonotonicResultsException(msg)
            prevScore = score
        return "max"
    elif allWrongScore > allCorrectScore:
        prevScore = allCorrectScore - 1
        for score in scoreList:
            if score > allWrongScore or score < allCorrectScore or score < prevScore:
                msg = "all-correct and all-incorrect trials indicated min optimality, "
                msg += "but mixed correct/incorrect scores were not monotonicly "
                msg += "decreasing with correctness"
                raise _NonMonotonicResultsException(msg)
            prevScore = score
        return "min"
    # allWrong and allCorrect must not be equal, otherwise it cannot be a measure
    # of correct performance
    else:
        msg = "functionToCheck produced the same values for trials including "
        msg += "all-correct and all-incorrect predictions."
        raise _NoDifferenceResultsException(msg)


def _generateAllZeros(length):
    return UML.createData("List", numpy.zeros([length, 1], dtype=int))


def _generateAllOnes(length):
    return UML.createData("List", numpy.ones([length, 1], dtype=int))


def _generateMixedRandom(length):
    while True:
        correct = numpyRandom.randint(2, size=[length, 1])
        # we don't want all zeros or all ones
        if numpy.any(correct) and not numpy.all(correct):
            break
    correct = UML.createData(returnType="List", data=correct)
    return correct


def _generatePredicted(knowns, predictionType):
    """
    Predicted may mean any of the three kinds of output formats for trainAndApply:
    predicted labels, bestScores, or allScores. If confidences are involved, they are
    randomly generated, yet consistent with correctness

    """
    workingCopy = knowns.copy()
    workingCopy.setFeatureName(0, 'PredictedClassLabel')
    # Labels
    if predictionType == 0:
        return workingCopy
    # Labels and the score for that label (aka 'bestScores')
    elif predictionType == 1:
        scores = numpyRandom.randint(2, size=[workingCopy.points, 1])
        scores = UML.createData(returnType="List", data=scores, featureNames=['LabelScore'])
        workingCopy.addFeatures(scores)
        return workingCopy
    # Labels, and scores for all possible labels (aka 'allScores')
    else:
        dataToFill = []
        for i in range(workingCopy.points):
            currConfidences = [None, None]
            winner = numpyRandom.randint(10) + 10 + 2
            loser = numpyRandom.randint(winner - 2) + 2
            if knowns.data[i][0] == 0:
                currConfidences[0] = winner
                currConfidences[1] = loser
            else:
                currConfidences[0] = loser
                currConfidences[1] = winner
            dataToFill.append(currConfidences)

        scores = UML.createData(returnType="List", data=dataToFill, featureNames=['0', '1'])
        return scores


def _makeIncorrect(predicted, predictionType, index):
    if predictionType in [0, 1]:
        predicted.data[index][0] = math.fabs(predicted.data[index][0] - 1)
    else:
        temp = predicted.data[index][0]
        predicted.data[index][0] = predicted.data[index][1]
        predicted.data[index][1] = temp


class _NonMonotonicResultsException(Exception):
    """
        Exception to be thrown if the results returned by a performance
        function do not trend monotoniclly up or down as predicted
        data approach known values.
    """

    def __init__(self, value):
        self.value = value

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

    def __str__(self):
        return repr(self.value)
