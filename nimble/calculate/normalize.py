"""
Normalize
"""
import functools

import numpy

import nimble

def meanNormalize(values1, values2=None):
    """
    Subtract the vector mean from each element.

    Normalization of ``values1`` is calculated by subtracting the mean
    of ``values1`` from each element. The mean of ``values1`` is also
    subtracted from each element in ``values2``, when applicable.

    Examples
    --------
    >>> raw1 = [[1], [2], [3], [4], [5]]
    >>> data1 = nimble.data('Matrix', raw1)
    >>> meanNormalize(data1)
    Matrix(
        [[-2.000]
         [-1.000]
         [0.000 ]
         [1.000 ]
         [2.000 ]]
        )
    >>> raw2 = [[3], [2], [6]]
    >>> data2 = nimble.data('Matrix', raw2)
    >>> norm1, norm2 = meanNormalize(data1, data2)
    >>> norm2
    Matrix(
        [[0.000 ]
         [-1.000]
         [3.000 ]]
        )
    """
    mean = nimble.calculate.mean(values1)

    meanNorm1 = values1 - mean
    if values2 is None:
        return meanNorm1

    meanNorm2 = values2 - mean

    return meanNorm1, meanNorm2

def zScoreNormalize(values1, values2=None):
    """
    Convert vector elements to a z-score.

    The z-score normalization of ``values1`` is calculated by
    subtracting the mean of ``values1`` from each element and dividing
    by the standard deviation of ``values1``. The mean and standard
    deviation of ``values1`` are also used to calculate the z-score for
    each element in ``values2``, when applicable.

    Examples
    --------
    >>> raw1 = [[1], [2], [3], [4], [5]]
    >>> data1 = nimble.data('Matrix', raw1)
    >>> zScoreNormalize(data1)
    Matrix(
        [[-1.414]
         [-0.707]
         [0.000 ]
         [0.707 ]
         [1.414 ]]
        )
    >>> raw2 = [[3], [2], [6]]
    >>> data2 = nimble.data('Matrix', raw2)
    >>> norm1, norm2 = zScoreNormalize(data1, data2)
    >>> norm2
    Matrix(
        [[0.000 ]
         [-0.707]
         [2.121 ]]
        )
    """
    standardDeviation = nimble.calculate.standardDeviation(values1, False)

    if standardDeviation == 0:
        return meanNormalize(values1, values2)

    if values2 is None:
        return meanNormalize(values1) / standardDeviation

    meanNorm1, meanNorm2 = meanNormalize(values1, values2)
    zScores1 = meanNorm1 / standardDeviation
    zScores2 = meanNorm2 / standardDeviation

    return zScores1, zScores2


def rangeNormalize(values1, values2=None, *, start, end):
    """
    Convert elements to an inclusive range from start to end.
    """
    minimum = nimble.calculate.minimum(values1)
    maximum = nimble.calculate.maximum(values1)

    spread = maximum - minimum
    subMin1 = values1 - minimum
    range1 = end * subMin1 - start * subMin1
    if spread:
        range1 /= spread
    range1 += start

    if values2 is None:
        return range1

    subMin2 = values2 - minimum
    range2 = end * subMin2 - start * subMin2
    if spread:
        range2 /= spread
    range2 += start

    return range1, range2


def minMaxNormalize(values1, values2=None):
    """
    Convert values to range of 0 to 1.

    For ``values1``, the formula ``(element - min) / (max - min)``
    will be applied to each element, where ``min`` is the minimum value
    in ``values1`` and ``max`` is the maximum value in ``values1``.
    The minimum and maximum values from ``values1`` are also applied to
    the calculation for ``values2``, when applicable.

    Examples
    --------
    >>> raw1 = [[1], [2], [3], [4], [5]]
    >>> data1 = nimble.data('Matrix', raw1)
    >>> minMaxNormalize(data1)
    Matrix(
        [[0.000]
         [0.250]
         [0.500]
         [0.750]
         [1.000]]
        )
    >>> raw2 = [[3], [2], [6]]
    >>> data2 = nimble.data('Matrix', raw2)
    >>> norm1, norm2 = minMaxNormalize(data1, data2)
    >>> norm2
    Matrix(
        [[0.500]
         [0.250]
         [1.250]]
        )
    """
    return rangeNormalize(values1, values2, start=0, end=1)


def percentileNormalize(values1, values2=None):
    """
    Convert elements to a percentile.

    The percentile is equal to ``(k - 1) / (n - 1)`` where ``k`` is the
    element rank in the sorted data and ``n`` is the number of values in
    the vector, providing elements a range of 0 to 1. When the data
    contains equalivent elements, the calculated percentile is the mean
    (equalivent to the median) of the percentiles calculated at each
    equal element.

    For elements in ``values2`` that are not in ``values1``, any
    elements within ``value1``'s range will be calculated using linear
    interpolation, any elements less than the minimum of ``values1``
    will be set to 0 and any elements greater than the maximum of
    ``values1`` will be set to 1.

    Examples
    --------
    >>> raw1 = [[1], [2], [4], [4], [5]]
    >>> data1 = nimble.data('Matrix', raw1)
    >>> percentileNormalize(data1)
    Matrix(
        [[0.000]
         [0.250]
         [0.625]
         [0.625]
         [1.000]]
        )
    >>> raw2 = [[3], [2], [6]]
    >>> data2 = nimble.data('Matrix', raw2)
    >>> norm1, norm2 = percentileNormalize(data1, data2)
    >>> norm2
    Matrix(
        [[0.438]
         [0.250]
         [1.000]]
        )
    """
    arr1 = values1.copy('numpyarray', outputAs1D=True).astype(float)
    percentiles = arr1.copy()

    # index locations of sorted values
    argsort = numpy.argsort(arr1)
    anyNan = any(map(numpy.isnan, arr1))
    if anyNan: # ignore index for nan
        epsilon = 1 / (len(arr1) - 2)
    else:
        epsilon = 1 / (len(arr1) - 1)
    for i, idx in enumerate(argsort):
        if not numpy.isnan(percentiles[idx]):
            percentiles[idx] = i * epsilon

    # adjust percentiles for duplicate values
    for v in numpy.unique(arr1):
        sameVal = arr1 == v
        samePercs = percentiles[sameVal]
        if len(samePercs) > 1:
            percentiles[sameVal] = sum(samePercs) / len(samePercs)

    values1Percentiles = values1.copy()
    if len(values1.points) == 1:
        values1Percentiles.points.transform(lambda _: percentiles,
                                            useLog=False)
    else:
        values1Percentiles.features.transform(lambda _: percentiles,
                                              useLog=False)

    if values2 is None:
        return values1Percentiles

    sortedArr1 = arr1[argsort]
    sortedPercs = percentiles[argsort]
    minimum = numpy.nanmin(sortedArr1)
    maximum = numpy.nanmax(sortedArr1)

    interpVal = functools.partial(_interpPercentile, minimum, maximum,
                                  sortedArr1, sortedPercs)
    interpVector = numpy.vectorize(interpVal)
    values2Percentiles = values2.copy()
    if len(values2.points) == 1:
        values2Percentiles.points.transform(interpVector, useLog=False)
    else:
        values2Percentiles.features.transform(interpVector, useLog=False)

    return values1Percentiles, values2Percentiles

def _interpPercentile(minimum, maximum, sortedValues, sortedPercentiles, val):
    if numpy.isnan(val):
        return val
    if val == minimum == maximum:
        return 0.5 # all values were equal
    if val <= minimum:
        return 0.0
    if val >= maximum:
        return 1.0
    idx = numpy.searchsorted(sortedValues, val) # binary search
    leftVal, rightVal = sortedValues[idx - 1:idx + 1]
    leftPerc, rightPerc = sortedPercentiles[idx - 1:idx + 1]
    if val == rightVal:
        return rightPerc
    fractionToNext = (val - leftVal) / (rightVal - leftVal)
    percDist = rightPerc - leftPerc
    return float(leftPerc + percDist * fractionToNext)
