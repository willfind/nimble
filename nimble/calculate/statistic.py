import math
import collections
import functools

import numpy

import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination, PackageException
from nimble.utility import ImportModule, dtypeConvert

scipy = ImportModule('scipy')

numericalTypes = (int, float, numpy.number)

def numericRequired(func):
    """
    Handles None return for functions that require numeric data.
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TypeError, ValueError):
            nan = numpy.nan
            if func.__name__ == 'quartiles':
                return (nan, nan, nan)
            return nan
    return wrapped

def proportionMissing(values):
    """
    The proportion of values in the vector that are missing.

    Calculate proportion of entries in 'values' iterator that are
    are None or NaN.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    Examples
    --------
    >>> raw = [1, 2, float('nan'), 4, float('nan')]
    >>> vector = nimble.createData('Matrix', raw)
    >>> proportionMissing(vector)
    0.4
    """
    numMissing = sum(1 for _ in values.iterateElements(only=match.missing))
    return numMissing / len(values)

def proportionZero(values):
    """
    The proportion of values in the vector that are equal to zero.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    Examples
    --------
    >>> raw = [0, 1, 2, 3]
    >>> vector = nimble.createData('Matrix', raw)
    >>> proportionZero(vector)
    0.25
    """
    numVals = len(values)
    if values.getTypeString() == 'Sparse':
        return (numVals - values.data.nnz) / numVals
    numZero = sum(1 for _ in values.iterateElements(only=match.zero))
    return numZero / numVals

def minimum(values):
    """
    The minimum value in a vector.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    See Also
    --------
    maximum

    Examples
    --------
    >>> raw = [0, 1, 2, float('nan')]
    >>> vector = nimble.createData('Matrix', raw)
    >>> minimum(vector)
    0
    """
    return _minmax(values, 'min')

def maximum(values):
    """
    The maximum value in a vector.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    See Also
    --------
    minimum

    Examples
    --------
    >>> raw = [0, 1, 2, float('nan')]
    >>> vector = nimble.createData('Matrix', raw)
    >>> maximum(vector)
    2
    """
    return _minmax(values, 'max')

@numericRequired
def _minmax(values, minmax):
    """
    Backend for finding the minimum or maximum value in a vector.
    """
    # convert to list not array b/c arrays won't error with non numeric data
    if values.getTypeString() == 'Sparse':
        toProcess = values.data.data.tolist()
        if len(values) > values.data.nnz:
            toProcess.append(0) # if sparse object has zeros add zero to list
    else:
        toProcess = values.copy('numpyarray')
    if minmax == 'min':
        ret = numpy.nanmin(toProcess)
    else:
        ret = numpy.nanmax(toProcess)

    if isinstance(ret, str):
        return None
    else:
        return ret

def _mean_sparseBackend(nonZeroVals, lenData, numNan):
    """
    Backend helper for sparse mean calculation. The number of nan values
    is needed for standard deviation calculations so this helper avoids
    repeated attempts to determine the number of nan values in the data.
    """
    dataSum = numpy.nansum(nonZeroVals)
    return dataSum / (lenData - numNan)

@numericRequired
def mean(values):
    """
    The mean of the values in a vector.

    This function requires numeric data and ignores any NaN values.
    Non-numeric values will results in None being returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    See Also
    --------
    median, mode

    Examples
    --------
    >>> raw = [0, 1, 2, float('nan'), float('nan'), 5]
    >>> vector = nimble.createData('Matrix', raw)
    >>> mean(vector)
    2.0
    """
    if values.getTypeString() == 'Sparse':
        nonZero = values.data.data.astype(numpy.float)
        numNan = numpy.sum(numpy.isnan(nonZero))
        return _mean_sparseBackend(nonZero, len(values), numNan)
    arr = values.copy('numpyarray', outputAs1D=True).astype(numpy.float)
    return numpy.nanmean(arr)

@numericRequired
def median(values):
    """
    The median of the values in a vector.

    This function requires numeric data and ignores any NaN values.
    Non-numeric values will results in None being returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    See Also
    --------
    mean, mode

    Examples
    --------
    >>> raw = [0, 1, 2, float('nan'), float('nan'), 5, 6]
    >>> vector = nimble.createData('Matrix', raw)
    >>> median(vector)
    2.0
    """
    arr = values.copy('numpyarray', outputAs1D=True).astype(numpy.float)
    return numpy.nanmedian(arr, overwrite_input=True)

def mode(values):
    """
    The mode of the values in a vector.

    This function requires numeric data and ignores any NaN values.
    Non-numeric values will results in None being returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    See Also
    --------
    mean, median

    Examples
    --------
    >>> raw = [0, 1, 2, float('nan'), float('nan'), float('nan'), 0, 6]
    >>> vector = nimble.createData('Matrix', raw)
    >>> mode(vector)
    0
    """
    if values.getTypeString() == 'Sparse':
        numZero = len(values) - values.data.nnz
        toProcess = values.iterateElements(only=nonMissingNonZero)
        counter = collections.Counter(toProcess)
        mcVal, mcCount = counter.most_common()[0]
        mostCommon = 0 if numZero > mcCount else mcVal
    else:
        toProcess = values.iterateElements(only=nonMissing)
        counter = collections.Counter(toProcess)
        mostCommon = counter.most_common()[0][0]
    return mostCommon

@numericRequired
def standardDeviation(values, sample=False):
    """
    The standard deviation of the values in a vector.

    This function requires numeric data and ignores any NaN values.
    Non-numeric values will results in None being returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.
    sample : bool
        If False, the default, the population standard deviation is
        returned. If True, the sample standard deviation is returned.

    Examples
    --------
    >>> raw = [1, 2, 3, 4, 5, 6, float('nan')]
    >>> vector = nimble.createData('Matrix', raw)
    >>> standardDeviation(vector)
    1.707825127659933
    >>> standardDeviation(vector, sample=True)
    1.8708286933869707
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

    arr = values.copy('numpyarray', outputAs1D=True).astype(numpy.float)
    if sample:
        return numpy.nanstd(arr, ddof=1)
    return numpy.nanstd(arr)

def uniqueCount(values):
    """
    The number of unique values in the vector.
    """
    if values.getTypeString() == 'Sparse':
        toProcess = values.iterateElements(only=nonMissingNonZero)
        valueSet = set(toProcess)
        if len(values) > values.data.nnz:
            valueSet.add(0)
    else:
        toProcess = values.iterateElements(only=nonMissing)
        valueSet = set(toProcess)
    return len(valueSet)

@numericRequired
def quartiles(values):
    """
    A vector's lower quartile, the median, and the upper quartile.

    Return a 3-tuple (lowerQuartile, median, upperQuartile). This
    function requires numeric data and ignores any NaN values.
    Non-numeric values will results in None being returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    See Also
    --------
    median

    Examples
    --------
    >>> raw = [1, 5, 12, 13, 14, 21, 23, float('nan')]
    >>> vector = nimble.createData('Matrix', raw)
    >>> quartiles(vector)
    (8.5, 13.0, 17.5)
    """
    # copy as horizontal array to use for intermediate calculations
    values = dtypeConvert(values.copy(to="numpyarray", outputAs1D=True))
    ret = numpy.nanpercentile(values, (25, 50, 75), overwrite_input=True)
    return tuple(ret)

def nonMissing(elem):
    return not match.missing(elem)

def nonMissingNonZero(e):
    return nonMissing(e) and match.nonZero(e)

def residuals(toPredict, controlVars):
    """
    Calculate the residuals by a linear regression model.

    Parameters
    ----------
    toPredict : nimble Base object
        Each feature will be used as the independent variable in a
        separate linear regression model with the ``controlVars`` as the
        dependent variables.
    controlVars : nimble Base object
        Must have the same number of points as toPredict. Each point
        will be used as the dependant variables to do predictions for
        the corresponding point in ``toPredict``.
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
    workingCV = dtypeConvert(workingCV.copy(to="numpy array"))
    workingTP = dtypeConvert(toPredict.copy(to="numpy array"))

    x,res,r,s = scipy.linalg.lstsq(workingCV, workingTP)
    pred = numpy.matmul(workingCV, x)
    ret = toPredict - pred
    return ret
