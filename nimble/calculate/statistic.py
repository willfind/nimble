from __future__ import division
from __future__ import absolute_import
import math

import numpy

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination, PackageException

scipy = nimble.importModule('scipy')

numericalTypes = (int, float, int, numpy.number)


def proportionMissing(values):
    """
    Calculate proportion of entries in 'values' iterator that are missing (defined
    as being None or NaN).
    """
    numMissing = 0
    numTotal = len(values)
    for value in values:
        if _isMissing(value):
            numMissing += 1
        else:
            pass

    if numTotal > 0:
        return float(numMissing) / float(numTotal)
    else:
        return 0.0


def proportionZero(values):
    """
    Calculate proportion of entries in 'values' iterator that are equal to zero.
    """
    totalNum = len(values)
    nonZeroCount = 0
    nonZeroItr = values.points.nonZeroIterator()
    for value in nonZeroItr:
        nonZeroCount += 1

    if totalNum > 0:
        return float(totalNum - nonZeroCount) / float(totalNum)
    else:
        return 0.0


def minimum(values, ignoreNoneNan=True, noCompMixType=True):
    """
    Given a 1D vector of values, find the minimum value.
    """
    return _minmax(values, 'min', ignoreNoneNan, noCompMixType)


def maximum(values, ignoreNoneNan=True, noCompMixType=True):
    """
    Given a 1D vector of values, find the maximum value.
    """
    return _minmax(values, 'max', ignoreNoneNan, noCompMixType)


def _minmax(values, minmax, ignoreNoneNan=True, noCompMixType=True):
    """
    Given a 1D vector of values, find the minimum or maximum value.
    """
    if minmax == 'min':
        compStr = '__lt__'
        func1 = lambda x, y: x > y
        func2 = lambda x, y: x < y
    else:
        compStr = '__gt__'
        func1 = lambda x, y: x < y
        func2 = lambda x, y: x > y

    first = True
    nonZeroValues = values.points.nonZeroIterator()
    count = 0

    #if data types are mixed and some data are not numerical, such as [1,'a']
    if noCompMixType and featureType(values) == 'Mixed' and not (_isNumericalFeatureGuesser(values)):
        return None
    for value in nonZeroValues:
        count += 1
        if ignoreNoneNan and _isMissing(value):
            continue
        if (hasattr(value, '__cmp__') or hasattr(value, compStr)):
            if first:
                currMin = value
                first = False
            else:
                if func2(value, currMin):
                    currMin = value

    if first:
        return None
    else:
        if func1(currMin, 0) and len(values) > count:
            return 0
        else:
            return currMin


def mean(values):
    """
    Given a 1D vector of values, find the mean value.  If the values are
    not numerical, return None.
    """
    if not _isNumericalFeatureGuesser(values):
        return None

    numericalCount = 0
    nonZeroCount = 0
    runningSum = 0
    totalCount = len(values)
    nonZeroValues = values.points.nonZeroIterator()

    for value in nonZeroValues:
        nonZeroCount += 1
        if _isNumericalPoint(value):
            runningSum += value
            numericalCount += 1

    if numericalCount == 0 and totalCount > nonZeroCount:
        return 0
    elif numericalCount == 0 and totalCount == nonZeroCount:
        return None
    elif numericalCount > 0 and totalCount == nonZeroCount:
        return float(runningSum) / float(numericalCount)
    elif numericalCount > 0 and totalCount > nonZeroCount:
        return float(runningSum) / float(numericalCount + totalCount - nonZeroCount)


def median(values):
    """
    Given a 1D vector of values, find the median value of the natural ordering.

    """
    if not _isNumericalFeatureGuesser(values):
        return None

    #Filter out None/NaN values from list of values
    sortedValues = [x for x in values if not _isMissing(x)]

    if len(sortedValues) == 0:
        return None

    sortedValues = sorted(sortedValues)

    numValues = len(sortedValues)

    if numValues % 2 == 0:
        median = (float(sortedValues[(numValues // 2) - 1]) + float(sortedValues[numValues // 2])) / float(2)
    else:
        median = float(sortedValues[int(math.floor(numValues / 2))])

    return median

def mode(values):
    """
    Given a 1D vector of values, find the most frequent value.
    """
    collections = nimble.importModule('collections')
    nonMissingValues = [x for x in values if not _isMissing(x)]
    counter = collections.Counter(nonMissingValues)
    return counter.most_common()[0][0]


def standardDeviation(values, sample=False):
    """
    Given a 1D vector of values, find the standard deviation.  If the values are
    not numerical, return None.
    """
    if not _isNumericalFeatureGuesser(values):
        return None

    #Filter out None/NaN values from list of values
    meanRet = mean(values)
    nonZeroCount = 0
    numericalCount = 0
    nonZeroValues = values.points.nonZeroIterator()

    squaredDifferenceTotal = 0
    for value in nonZeroValues:
        nonZeroCount += 1
        if _isNumericalPoint(value):
            numericalCount += 1
            squaredDifferenceTotal += (meanRet - value) ** 2

    if nonZeroCount < len(values):
        numZeros = len(values) - nonZeroCount
        squaredDifferenceTotal += numZeros * meanRet ** 2
        numericalCount += numZeros

    # doing sample covariance calculation
    if sample:
        divisor = numericalCount - 1
    # doing population covariance calculation
    else:
        divisor = numericalCount

    if divisor == 0:
        return 0

    stDev = math.sqrt(squaredDifferenceTotal / float(divisor))
    return stDev


def uniqueCount(values):
    """
    Given a 1D vector of values, calculate the number of unique values.
    """
    values = [x for x in values if not _isMissing(x)]
    valueSet = set(values)
    return len(valueSet)


def featureType(values):
    """
        Return the type of data: string, int, float
    """

    # types = numpy.unique([type(value) for value in values if not _isMissing(value)])#doesn't work in python3
    types = list(set([type(value) for value in values if not _isMissing(value)]))
    #if all data in values are missing
    if len(types) == 0:
        return 'Unknown'
    #if multiple types are in values
    elif len(types) > 1:
        return 'Mixed'
    else:
        return str(types[0]).split("'")[1]#in python2, it is like "<type 'xxx'>", while in python3, it is like "<class 'xxx'>"


def quartiles(values, ignoreNoneOrNan=True):
    """
    From the vector of values, return a 3-tuple containing the
    lower quartile, the median, and the upper quartile.

    """
    if not _isNumericalFeatureGuesser(values):
        return (None, None, None)

    if isinstance(values, nimble.data.Base):
        #conver to a horizontal array
        values = values.copy(to="numpyarray").flatten()

    if ignoreNoneOrNan:
        values = [v for v in values if not _isMissing(v)]
    ret = numpy.percentile(values, (25, 50, 75))

    return tuple(ret)


def _isMissing(point):
    """
    Determine if a point is missing or not.  If the point is None or NaN, return True.
    Else return False.
    """
    #this might be the fastest way
    return (point is None) or (point != point)


def _isNumericalFeatureGuesser(featureVector):
    """
    Returns true if the vector only contains primitive numerical non-complex values,
    returns false otherwise.
    """
    try:
        if featureVector.getTypeString() in ['Matrix']:
            return True
    except AttributeError:
        pass

    #if all items in featureVector are numerical or None/NaN, return True; otherwise, False.
    return all([isinstance(item, numericalTypes) for item in featureVector if item])


def _isNumericalPoint(point):
    """
    Check to see if a point is a valid number that can be used in numerical calculations.
    If point is of type float, long, or int, and not None or NaN, return True.  Otherwise
    return False.
    """
    #np.nan is in numericalTypes, but None isn't; None==None, but np.nan!=np.nan
    if isinstance(point, numericalTypes) and (point == point):
        return True
    else:
        return False


def residuals(toPredict, controlVars):
    """
    Calculate the residuals of toPredict, by a linear regression model using the controlVars.

    toPredict: nimble Base object, where each feature will be used as the independant
    variable in a separate linear regression model with the controlVars as the
    dependant variables.

    controlVars: nimble Base object, with the same number of points as toPredict. Each
    point will be used as the dependant variables to do predictions for the
    corresponding point in toPredict.

    Returns: nimble Base object of the same size as toPredict, containing the calculated
    residuals.

    Raises: InvalidArgumentType if toPredict and controlVars are not nimble
    data objects and InvalidArgumentValue if either has nonzero points
    or features and InvalidArgumentValueCombination if they have a different
    number of points.
    """
    if scipy is None:
        msg = "scipy must be installed in order to use the residuals function."
        raise PackageException(msg)

    if not isinstance(toPredict, nimble.data.Base):
        msg = "toPredict must be a nimble data object"
        raise InvalidArgumentType(msg)
    if not isinstance(controlVars, nimble.data.Base):
        msg = "controlVars must be a nimble data object"
        raise InvalidArgumentType(msg)

    tpP = len(toPredict.points)
    tpF = len(toPredict.features)
    cvP = len(controlVars.points)
    cvF = len(controlVars.features)

    if tpP != cvP:
        msg = "toPredict and controlVars must have the same number of points: ("
        msg += str(tpP) + ") vs (" + str(cvP) + ")"
        raise InvalidArgumentValueCombination(msg)
    if tpP == 0 or tpF == 0:
        msg = "toPredict must have nonzero points (" + str(tpP) + ") and "
        msg += "nonzero features (" + str(tpF) + ")"
        raise InvalidArgumentValue(msg)
    if cvP == 0 or cvF == 0:
        msg = "controlVars must have nonzero points (" + str(cvP) + ") and "
        msg += "nonzero features (" + str(cvF) + ")"
        raise InvalidArgumentValue(msg)

    workingType = controlVars.getTypeString()
    workingCV = controlVars.copy()
    workingCV.features.append(nimble.ones(workingType, cvP, 1), useLog=False)
    workingCV = workingCV.copy(to="numpy array")
    workingTP = toPredict.copy(to="numpy array")

    x,res,r,s = scipy.linalg.lstsq(workingCV, workingTP)
    pred = numpy.matmul(workingCV, x)
    ret = toPredict - pred
    return ret
