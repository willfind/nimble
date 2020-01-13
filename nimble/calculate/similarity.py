
import itertools

import numpy

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.calculate import fractionIncorrect
from nimble.calculate import varianceFractionRemaining
from nimble.data.dataHelpers import createDataNoValidation


def _validatePredictedAsLabels(predictedValues):
    if not isinstance(predictedValues, nimble.data.Base):
        msg = "predictedValues must be derived class of nimble.data.Base"
        raise InvalidArgumentType(msg)
    if len(predictedValues.features) > 1:
        msg = "predictedValues must be labels only; this has more than "
        msg += "one feature"
        raise InvalidArgumentValue(msg)


def cosineSimilarity(knownValues, predictedValues):
    _validatePredictedAsLabels(predictedValues)
    if not isinstance(knownValues, nimble.data.Base):
        msg = "knownValues must be derived class of nimble.data.Base"
        raise InvalidArgumentType(msg)

    known = knownValues.copy(to="numpy array").flatten()
    predicted = predictedValues.copy(to="numpy array").flatten()

    numerator = (numpy.dot(known, predicted))
    denominator = (numpy.linalg.norm(known) * numpy.linalg.norm(predicted))

    return numerator / denominator


cosineSimilarity.optimal = 'max'


def correlation(X, X_T=None):
    """
    Calculate the correlation between points in X. If X_T is not
    provided, a copy of X will be made in this function.

    """
    if X_T is None:
        X_T = X.T
    stdVector = X.points.statistics('populationstd')
    stdVector_T = stdVector.T

    cov = covariance(X, X_T, False)
    stdMatrix = stdVector.matrixMultiply(stdVector_T)
    ret = cov / stdMatrix

    return ret


def covariance(X, X_T=None, sample=True):
    """
    Calculate the covariance between points in X. If X_T is not
    provided, a copy of X will be made in this function.

    """
    if X_T is None:
        X_T = X.T
    pointMeansVector = X.points.statistics('mean')
    fill = lambda x: [x[0]] * len(X.features)
    pointMeans = pointMeansVector.points.calculate(fill, useLog=False)
    pointMeans_T = pointMeans.T

    XminusEofX = X - pointMeans
    X_TminusEofX_T = X_T - pointMeans_T

    # doing sample covariance calculation
    if sample:
        divisor = len(X.features) - 1
    # doing population covariance calculation
    else:
        divisor = len(X.features)

    ret = (XminusEofX.matrixMultiply(X_TminusEofX_T)) / divisor
    return ret


def fractionCorrect(knownValues, predictedValues):
    """
    Calculate how many values in predictedValues are equal to the
    values in the corresponding positions in knownValues. The return
    will be a float between 0 and 1 inclusive.

    """
    return 1 - fractionIncorrect(knownValues, predictedValues)


fractionCorrect.optimal = 'max'


def rSquared(knownValues, predictedValues):
    """
    Calculate the r-squared (or coefficient of determination) of the
    predictedValues given the knownValues. This will be equal to 1 -
    nimble.calculate.varianceFractionRemaining() of the same inputs.

    """
    return 1.0 - varianceFractionRemaining(knownValues, predictedValues)


rSquared.optimal = 'max'


def confusionMatrix(knownValues, predictedValues, labels=None,
                    convertCountsToFractions=False):
    """
    Generate a confusion matrix for known and predicted label values.

    The confusion matrix contains the counts of observations that
    occurred for each known/predicted pair. Features represent the
    known classes and points represent the predicted classes.
    Optionally, these can be output as fractions instead of counts.

    Parameters
    ----------
    knownValues : nimble Base object
        The ground truth labels collected for some data.
    predictedValues : nimble Base object
        The labels predicted for the same data.
    labels : dict, list
        As a dictionary, a mapping of from the value in ``knownLabels``
        to a more specific label. A list may also be used provided the
        values in ``knownLabels`` represent an index to each value in
        the list. The labels will be used to create the featureNames and
        pointNames with the prefixes "known_" and "predicted_",
        respectively.  If labels is None, the prefixes will be applied
        directly to the unique values found in ``knownLabels``.
    convertCountsToFractions : bool
        If False, the default, elements are counts. If True, the counts
        are converted to fractions by dividing by the total number of
        observations.

    Returns
    -------
    Base
        A confusion matrix nimble object matching the type of
        ``knownValues``.

    Notes
    -----
    Metrics for binary classification based on a confusion matrix,
    like truePositive, recall, precision, etc., can also be found in
    the nimble.calculate module.

    Examples
    --------
    Confusion matrix with and without alternate labels.

    >>> known = [[0], [1], [2],
    ...          [0], [1], [2],
    ...          [0], [1], [2],
    ...          [0], [1], [2]]
    >>> pred = [[0], [1], [2],
    ...         [0], [1], [2],
    ...         [0], [1], [2],
    ...         [1], [0], [2]]
    >>> knownObj = nimble.createData('Matrix', known)
    >>> predObj = nimble.createData('Matrix', pred)
    >>> cm = confusionMatrix(knownObj, predObj)
    >>> print(cm)
                  known_0 known_1 known_2
    <BLANKLINE>
    predicted_0      3       1       0
    predicted_1      1       3       0
    predicted_2      0       0       4
    <BLANKLINE>
    >>> labels = {0: 'cat', 1: 'dog', 2: 'fish'}
    >>> cm = confusionMatrix(knownObj, predObj, labels=labels)
    >>> print(cm)
                     known_cat known_dog known_fish
    <BLANKLINE>
     predicted_cat       3         1         0
     predicted_dog       1         3         0
    predicted_fish       0         0         4
    <BLANKLINE>

    Label objects can have string values and here we output fractions.

    >>> known = [['cat'], ['dog'], ['fish'],
    ...          ['cat'], ['dog'], ['fish'],
    ...          ['cat'], ['dog'], ['fish'],
    ...          ['cat'], ['dog'], ['fish']]
    >>> pred = [['cat'], ['dog'], ['fish'],
    ...         ['cat'], ['dog'], ['fish'],
    ...         ['cat'], ['dog'], ['fish'],
    ...         ['dog'], ['cat'], ['fish']]
    >>> knownObj = nimble.createData('Matrix', known)
    >>> predObj = nimble.createData('Matrix', pred)
    >>> cm = confusionMatrix(knownObj, predObj,
    ...                      convertCountsToFractions=True)
    >>> print(cm)
                     known_cat known_dog known_fish
    <BLANKLINE>
     predicted_cat     0.250     0.083     0.000
     predicted_dog     0.083     0.250     0.000
    predicted_fish     0.000     0.000     0.333
    <BLANKLINE>
    """
    if not (isinstance(knownValues, nimble.data.Base)
            and isinstance(predictedValues, nimble.data.Base)):
        msg = 'knownValues and predictedValues must be nimble data objects'
        raise InvalidArgumentType(msg)
    if not knownValues.shape[1] == predictedValues.shape[1] == 1:
        msg = 'knownValues and predictedValues must each be a single feature'
        raise InvalidArgumentValue(msg)
    if knownValues.shape[0] != predictedValues.shape[0]:
        msg = 'knownValues and predictedValues must have the same number of '
        msg += 'points'
        raise InvalidArgumentValue(msg)
    if not isinstance(labels, (type(None), dict, list)):
        msg = 'labels must be a dictionary mapping values from knownValues to '
        msg += 'a label or a list if the unique values in knownValues are in '
        msg += 'the range 0 to len(labels)'
        raise InvalidArgumentType(msg)

    numPts = len(knownValues.points)
    labelsProvided = labels is not None

    mappedLabels = {} # cache mapped labels
    def mapInt(val):
        try:
            return mappedLabels[val]
        except KeyError:
            try:
                if val % 1 == 0:
                    mappedLabels[val] = int(val)
                    return int(val)
                return val
            except TypeError:
                return val

    knownLabels = set()
    confusionDict = {}
    for kVal, pVal in zip(knownValues.elements, predictedValues.elements):
        if labelsProvided:
            try:
                kVal = labels[mapInt(kVal)]
            except KeyError:
                msg = '{kVal} is not a valid key for the labels argument'
                raise KeyError(msg.format(kVal=kVal))
            except IndexError:
                msg = '{kVal} is not a valid index for the labels argument'
                raise IndexError(msg.format(kVal=mapInt(kVal)))
            # safe to assume predicted labels will be one of known labels?
            pVal = labels[mapInt(pVal)]

        knownLabels.add(kVal)
        if (kVal, pVal) in confusionDict:
            confusionDict[(kVal, pVal)] += 1
        else:
            confusionDict[(kVal, pVal)] = 1

    if labels is None:
        knownLabels = sorted(list(map(mapInt, knownLabels)))
    elif isinstance(labels, dict):
        knownLabels = [labels[key] for key in sorted(labels)]
    else:
        knownLabels = labels

    length = len(knownLabels)
    if convertCountsToFractions:
        dtype = float
    else:
        dtype = int

    confusionMtx = numpy.zeros((length, length), dtype=dtype)

    for pInfo, kInfo in itertools.product(enumerate(knownLabels), repeat=2):
        pIdx, pLbl = pInfo
        kIdx, kLbl = kInfo
        try:
            value = confusionDict[(kLbl, pLbl)]
            if convertCountsToFractions:
                value /= numPts
            confusionMtx[pIdx, kIdx] = value
        except KeyError:
            pass # never predicted keep value as zero at this index

    asType = knownValues.getTypeString()
    fNames = ['known_' + str(label) for label in knownLabels]
    pNames = ['predicted_' + str(label) for label in knownLabels]

    return createDataNoValidation(asType, confusionMtx, pNames, fNames,
                                  reuseData=True)
