
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
Statistics calculations.
"""

import collections
import functools

import numpy as np

import nimble
from nimble import match
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination, PackageException
from nimble._utility import scipy
from nimble._utility import dtypeConvert

numericalTypes = (int, float, np.number)

# alias for python sum since we define our own sum function here
_sum = sum

def numericRequired(func):
    """
    Handles NaN return for functions that require numeric data.
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TypeError, ValueError):
            nan = np.nan
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
    >>> lst = [1, 2, float('nan'), 4, float('nan')]
    >>> vector = nimble.data(lst)
    >>> proportionMissing(vector)
    0.4
    """
    numMissing = _sum(1 for _ in values.iterateElements(only=match.missing))
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
    >>> lst = [0, 1, 2, 3]
    >>> vector = nimble.data(lst)
    >>> proportionZero(vector)
    0.25
    """
    numVals = len(values)
    if values.getTypeString() == 'Sparse':
        return (numVals - values._data.nnz) / numVals
    numZero = _sum(1 for _ in values.iterateElements(only=match.zero))
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
    >>> lst = [0, 1, 2, float('nan')]
    >>> vector = nimble.data(lst)
    >>> minimum(vector)
    0.0
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
    >>> lst = [0, 1, 2, float('nan')]
    >>> vector = nimble.data(lst)
    >>> maximum(vector)
    2.0
    """
    return _minmax(values, 'max')

@numericRequired
def _minmax(values, minmax):
    """
    Backend for finding the minimum or maximum value in a vector.
    """
    #import pdb; pdb.set_trace()
    # convert to list not array b/c arrays won't error with non numeric data
    if values.getTypeString() == 'Sparse':
        toProcess = values._data.data.tolist()
        if len(values) > values._data.nnz:
            toProcess.append(0) # if sparse object has zeros add zero to list
    else:
        toProcess = values.copy('numpyarray')
        if toProcess.dtype == np.object_:
            if any(not isinstance(v, (np.int_, np.bool_))
                   for v in toProcess):
                toProcess = toProcess.astype(float)
            else:
                toProcess = toProcess.astype(int)
    if minmax == 'min':
        ret = np.nanmin(toProcess)
    else:
        ret = np.nanmax(toProcess)

    if isinstance(ret, str):
        return np.nan

    return ret

def _meanSparseBackend(nonZeroVals, lenData, numNan):
    """
    Backend helper for sparse mean calculation. The number of nan values
    is needed for standard deviation calculations so this helper avoids
    repeated attempts to determine the number of nan values in the data.
    """
    dataSum = np.nansum(nonZeroVals)
    return dataSum / (lenData - numNan)

@numericRequired
def mean(values):
    """
    The mean of the values in a vector.

    This function requires numeric data and ignores any NaN values.
    Non-numeric values will results in NaN being returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    See Also
    --------
    median, mode

    Examples
    --------
    >>> lst = [0, 1, 2, float('nan'), float('nan'), 5]
    >>> vector = nimble.data(lst)
    >>> mean(vector)
    2.0
    """
    #import pdb; pdb.set_trace()
    if values.getTypeString() == 'Sparse':
        nonZero = values._data.data.astype(np.float64)
        numNan = np.sum(np.isnan(nonZero))
        return _meanSparseBackend(nonZero, len(values), numNan)
    arr = values.copy('numpyarray', outputAs1D=True).astype(np.float64)
    return np.nanmean(arr)

@numericRequired
def median(values):
    """
    The median of the values in a vector.

    This function requires numeric data and ignores any NaN values.
    Non-numeric values will results in NaN being returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    See Also
    --------
    mean, mode

    Examples
    --------
    >>> lst = [0, 1, 2, float('nan'), float('nan'), 5, 6]
    >>> vector = nimble.data(lst)
    >>> median(vector)
    2.0
    """
    arr = values.copy('numpyarray', outputAs1D=True).astype(np.float64)
    return np.nanmedian(arr, overwrite_input=True)

def mode(values):
    """
    The mode of the values in a vector.

    When the mode can be more than one value, those values are sorted
    and the middle value (for an odd number of mode values) or first
    value to the left of the middle (for an even number of mode values)
    will be returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    See Also
    --------
    mean, median

    Examples
    --------
    >>> lst = [0, 1, 2, float('nan'), float('nan'), float('nan'), 0, 6]
    >>> vector = nimble.data(lst)
    >>> mode(vector)
    0.0
    """
    if values.getTypeString() == 'Sparse':
        numZero = len(values) - values._data.nnz
        toProcess = values.iterateElements(only=nonMissingNonZero)
        counter = collections.Counter(toProcess)
        if numZero:
            counter.update({0: numZero})
    else:
        toProcess = values.iterateElements(only=nonMissing)
        counter = collections.Counter(toProcess)
    if not counter:
        msg = 'The mode could not be calculated, the object either contained '
        msg += 'no values or all values were missing'
        raise InvalidArgumentValue(msg)
    mostCommon = counter.most_common()
    first, modeCount = mostCommon[0]
    modes = [first]
    for val, cnt in mostCommon[1:]:
        if cnt == modeCount:
            modes.append(val)
        else:
            break

    numModes = len(modes)
    if numModes == 1:
        return modes[0]
    center = numModes // 2
    if not numModes % 2:
        center -= 1 # use left of the two center values

    return sorted(modes)[center]

@numericRequired
def variance(values, sample=True):
    """
    The variance of the values in a vector.

    This function requires numeric data and ignores any NaN values.
    Non-numeric values will results in NaN being returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.
    sample : bool
        If True, the default, the sample standard deviation is
        returned. If False, the population standard deviation is returned.

    Examples
    --------
    >>> lst = [1, 2, 3, 4, 5, 6, float('nan')]
    >>> vector = nimble.data(lst)
    >>> variance(vector)
    3.5
    >>> variance(vector, sample=False)
    2.9166666666666665
    """
    if values.getTypeString() == 'Sparse':
        nonZero = values._data.data.astype(np.float64)
        numNan = np.sum(np.isnan(nonZero))
        meanRet = _meanSparseBackend(nonZero, len(values), numNan)

        dataSumSquared = np.nansum((nonZero - meanRet) ** 2)
        zeroSumSquared = meanRet ** 2 * (len(values) - values._data.nnz)
        divisor = len(values) - numNan
        if sample:
            divisor -= 1
        return (dataSumSquared + zeroSumSquared) / divisor

    arr = values.copy('numpyarray', outputAs1D=True).astype(np.float64)
    if sample:
        return np.nanvar(arr, ddof=1)
    return np.nanvar(arr)

@numericRequired
def standardDeviation(values, sample=True):
    """
    The standard deviation of the values in a vector.

    This function requires numeric data and ignores any NaN values.
    Non-numeric values will results in NaN being returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.
    sample : bool
        If True, the default, the sample standard deviation is
        returned. If False, the population standard deviation is returned.

    Examples
    --------
    >>> lst = [1, 2, 3, 4, 5, 6, float('nan')]
    >>> vector = nimble.data(lst)
    >>> standardDeviation(vector)
    1.8708286933869707
    >>> standardDeviation(vector, sample=False)
    1.707825127659933
    """
    return np.sqrt(variance(values, sample))

@numericRequired
def medianAbsoluteDeviation(values):
    """
    The median absolute deviation of the values in a vector.

    This function requires numeric data and ignores any NaN values.
    Non-numeric values will results in NaN being returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    Examples
    --------
    >>> lst = [1, 2, 3, 4, 5, 6, float('nan')]
    >>> vector = nimble.data(lst)
    >>> medianAbsoluteDeviation(vector)
    1.5
    """
    arr = values.copy('numpy array', outputAs1D=True).astype(np.float64)
    mad = np.nanmedian(np.abs(arr - np.nanmedian(arr)))
    return mad

def uniqueCount(values):
    """
    The number of unique values in the vector.
    """
    if values.getTypeString() == 'Sparse':
        toProcess = values.iterateElements(only=nonMissingNonZero)
        valueSet = set(toProcess)
        if len(values) > values._data.nnz:
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
    Non-numeric values will results in NaN being returned.

    Parameters
    ----------
    values : nimble Base object
        Must be one-dimensional.

    See Also
    --------
    median

    Examples
    --------
    >>> lst = [1, 5, 12, 13, 14, 21, 23, float('nan')]
    >>> vector = nimble.data(lst)
    >>> quartiles(vector)
    (8.5, 13.0, 17.5)
    """
    # copy as horizontal array to use for intermediate calculations
    values = dtypeConvert(values.copy(to="numpyarray", outputAs1D=True))
    ret = np.nanpercentile(values, (25, 50, 75), overwrite_input=True)
    return tuple(ret)

def nonMissing(elem):
    """
    True for any non-missing element, otherwise False.
    """
    return not match.missing(elem)

def nonMissingNonZero(elem):
    """
    True for any non-missing and non-zero element, otherwise False.
    """
    return nonMissing(elem) and match.nonZero(elem)

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
    if not scipy.nimbleAccessible():
        msg = "scipy must be installed in order to use the residuals function."
        raise PackageException(msg)

    if not isinstance(toPredict, nimble.core.data.Base):
        msg = "toPredict must be a nimble data object"
        raise InvalidArgumentType(msg)
    if not isinstance(controlVars, nimble.core.data.Base):
        msg = "controlVars must be a nimble data object"
        raise InvalidArgumentType(msg)

    tpP = len(toPredict.points)
    tpF = len(toPredict.features)
    cvP = len(controlVars.points)
    cvF = len(controlVars.features)

    if tpP != cvP:
        msg = "toPredict and controlVars must have the same number of points: "
        msg += "(" + str(tpP) + ") vs (" + str(cvP) + ")"
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
    workingCV.features.append(nimble.ones(cvP, 1, returnType=workingType),
                              useLog=False)
    workingCV = dtypeConvert(workingCV.copy(to="numpy array"))
    workingTP = dtypeConvert(toPredict.copy(to="numpy array"))

    x, _, _, _ = scipy.linalg.lstsq(workingCV, workingTP)
    pred = np.matmul(workingCV, x)
    ret = toPredict - pred
    return ret

def count(values):
    """
    The number of values in the vector.

    This function allows for non-numeric values but ignores any NaN
    values.
    """
    return _sum(1 for _ in values.iterateElements(only=nonMissing))

def sum(values): # pylint: disable=redefined-builtin
    """
    The sum of the values in the vector.

    This function requires numeric data and ignores any NaN values.
    Non-numeric values will results in NaN being returned.
    """
    return _sum(v for v in values.iterateElements(only=nonMissingNonZero))
