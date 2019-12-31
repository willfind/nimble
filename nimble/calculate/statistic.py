import math
import collections
import functools

import numpy

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination, PackageException
from nimble.utility import ImportModule

scipy = ImportModule('scipy')

numericalTypes = (int, float, numpy.number)

def numericRequired(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TypeError, ValueError):
            if func.__name__ == 'quartiles':
                return (None, None, None)
            return None
    return wrapped

def proportionMissing(values):
    """
    Calculate proportion of entries in 'values' iterator that are
    missing (defined as being None or NaN).
    """
    numMissing = 0
    numTotal = len(values)
    for value in values.points.nonZeroIterator():
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
    Calculate proportion of entries in 'values' iterator that are equal
    to zero.
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

def minimum(values, ignoreNoneNan=True):
    """
    Given a 1D vector of values, find the minimum value.
    """
    return _minmax(values, 'min', ignoreNoneNan)

def maximum(values, ignoreNoneNan=True):
    """
    Given a 1D vector of values, find the maximum value.
    """
    return _minmax(values, 'max', ignoreNoneNan)

@numericRequired
def _minmax(values, minmax, ignoreNoneNan):
    """
    Given a 1D vector of values, find the minimum or maximum value.
    """
    # convert to list not array b/c arrays can handle non numeric data
    if values.getTypeString() == 'Sparse':
        lst = values.data.data.tolist()
        if len(values) > values.data.nnz:
            lst.append(0) # if sparse object has zeros add zero to list
    else:
        lst = values.copy('pythonlist')
    if minmax == 'min' and ignoreNoneNan:
        return numpy.nanmin(lst)
    if minmax == 'min':
        return numpy.min(lst)
    if minmax == 'max' and ignoreNoneNan:
        return numpy.nanmax(lst)
    else:
        return numpy.max(lst)

def _mean_sparseBackend(nonZeroVals, lenData, numNan):
    dataSum = numpy.nansum(nonZeroVals)
    return dataSum / (lenData - numNan)

@numericRequired
def mean(values):
    """
    Given a 1D vector of values, find the mean value.  If the values are
    not numerical, return None.
    """
    if values.getTypeString() == 'Sparse':
        nonZero = values.data.data.astype(numpy.float)
        numNan = numpy.sum(numpy.isnan(nonZero))
        return _mean_sparseBackend(nonZero, len(values), numNan)
    arr = values.copy('numpyarray').astype(numpy.float)
    return numpy.nanmean(arr)

@numericRequired
def median(values):
    """
    Given a 1D vector of values, find the median value of the natural
    ordering.
    """
    arr = values.copy('numpyarray').astype(numpy.float)
    return numpy.nanmedian(arr)

def mode(values):
    """
    Given a 1D vector of values, find the most frequent value.
    """
    nonMissingValues = [x for x in values if not _isMissing(x)]
    counter = collections.Counter(nonMissingValues)
    return counter.most_common()[0][0]

@numericRequired
def standardDeviation(values, sample=False):
    """
    Given a 1D vector of values, find the standard deviation.  If the
    values are not numerical, return None.
    """
    if values.getTypeString() == 'Sparse':
        nonZero = values.data.data.astype(numpy.float)
        numNan = numpy.sum(numpy.isnan(nonZero))
        meanRet = _mean_sparseBackend(nonZero, len(values), numNan)

        dataSumSquared = numpy.nansum((nonZero - meanRet) ** 2)
        zeroSumSquared = meanRet ** 2 * (len(values) - values.data.nnz)
        divisor = len(values) - numNan
        if sample:
            divisor -= 1
        var = (dataSumSquared + zeroSumSquared) / divisor
        return numpy.sqrt(var)

    arr = values.copy('numpyarray').astype(numpy.float)
    if sample:
        return numpy.nanstd(arr, ddof=1)
    return numpy.nanstd(arr)

def uniqueCount(values):
    """
    Given a 1D vector of values, calculate the number of unique values.
    """
    values = [x for x in values if not _isMissing(x)]
    valueSet = set(values)
    return len(valueSet)


@numericRequired
def quartiles(values, ignoreNoneOrNan=True):
    """
    From the vector of values, return a 3-tuple containing the
    lower quartile, the median, and the upper quartile.
    """
    if isinstance(values, nimble.data.Base):
        #conver to a horizontal array
        values = values.copy(to="numpyarray").flatten()

    if ignoreNoneOrNan:
        values = [v for v in values if not _isMissing(v)]
    ret = numpy.percentile(values, (25, 50, 75))

    return tuple(ret)


def _isMissing(value):
    """
    Determine if a point is missing or not.  If the point is None or NaN, return True.
    Else return False.
    """
    #this might be the fastest way
    return (value is None) or (value != value)


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
    if not scipy:
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
