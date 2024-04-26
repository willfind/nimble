
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
Normalize
"""
import functools

import numpy as np

import nimble

def meanNormalize(values1, values2=None):
    """
    Subtract the vector mean from each element.

    Normalization of ``values1`` is calculated by subtracting the mean
    of ``values1`` from each element. The mean of ``values1`` is also
    subtracted from each element in ``values2``, when applicable. This
    normalization is also known as "Centered" or "Centering".

    Examples
    --------
    >>> lst1 = [[1], [2], [3], [4], [5]]
    >>> X1 = nimble.data(lst1)
    >>> meanNormalize(X1)
    <Matrix 5pt x 1ft
           0
       ┌───────
     0 │ -2.000
     1 │ -1.000
     2 │  0.000
     3 │  1.000
     4 │  2.000
    >
    >>> lst2 = [[3], [2], [6]]
    >>> X2 = nimble.data(lst2)
    >>> norm1, norm2 = meanNormalize(X1, X2)
    >>> norm2
    <Matrix 3pt x 1ft
           0
       ┌───────
     0 │  0.000
     1 │ -1.000
     2 │  3.000
    >
    """
    mean = nimble.calculate.mean(values1)

    meanNorm1 = values1 - mean
    if values2 is None:
        return meanNorm1

    meanNorm2 = values2 - mean

    return meanNorm1, meanNorm2

def meanStandardDeviationNormalize(values1, values2=None):
    """
    Subtract the mean and divide by standard deviation for each element.

    The normalization of ``values1`` is calculated by subtracting its
    mean and dividing by its standard deviation. The mean and standard
    deviation of ``values1`` are also used for the calculation on each
    element in ``values2``, when applicable. This normalization is also
    known as "Standardization" and "Z-score normalization".

    Examples
    --------
    >>> lst1 = [[1], [2], [3], [4], [5]]
    >>> X1 = nimble.data(lst1)
    >>> meanStandardDeviationNormalize(X1)
    <Matrix 5pt x 1ft
           0
       ┌───────
     0 │ -1.414
     1 │ -0.707
     2 │  0.000
     3 │  0.707
     4 │  1.414
    >
    >>> lst2 = [[3], [2], [6]]
    >>> X2 = nimble.data(lst2)
    >>> norm1, norm2 = meanStandardDeviationNormalize(X1, X2)
    >>> norm2
    <Matrix 3pt x 1ft
           0
       ┌───────
     0 │  0.000
     1 │ -0.707
     2 │  2.121
    >
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


def range0to1Normalize(values1, values2=None):
    """
    Convert values to range of 0 to 1.

    For ``values1``, the formula ``(element - min) / (max - min)``
    will be applied to each element, where ``min`` is its minimum value
    and ``max`` is its maximum value. The minimum and maximum values
    from ``values1`` are also applied to the calculation for
    ``values2``, when applicable. This normalization is often referred
    to simply as "Normalization".

    Examples
    --------
    >>> lst1 = [[1], [2], [3], [4], [5]]
    >>> X1 = nimble.data(lst1)
    >>> range0to1Normalize(X1)
    <Matrix 5pt x 1ft
           0
       ┌──────
     0 │ 0.000
     1 │ 0.250
     2 │ 0.500
     3 │ 0.750
     4 │ 1.000
    >
    >>> lst2 = [[3], [2], [6]]
    >>> X2 = nimble.data(lst2)
    >>> norm1, norm2 = range0to1Normalize(X1, X2)
    >>> norm2
    <Matrix 3pt x 1ft
           0
       ┌──────
     0 │ 0.500
     1 │ 0.250
     2 │ 1.250
    >
    """
    return rangeNormalize(values1, values2, start=0, end=1)


def percentileNormalize(values1, values2=None):
    """
    Convert elements to a percentile.

    The percentile is equal to ``(k - 1) / (n - 1)`` where ``k`` is the
    element rank in the sorted data and ``n`` is the number of values in
    the vector, providing elements a range of 0 to 1. When the data
    contains equivalent elements, the calculated percentile is the mean
    (equivalent to the median) of the percentiles calculated at each
    equal element.

    For elements in ``values2`` that are not in ``values1``, any
    elements within ``value1``'s range will be calculated using linear
    interpolation, any elements less than the minimum of ``values1``
    will be set to 0 and any elements greater than the maximum of
    ``values1`` will be set to 1.

    Examples
    --------
    >>> lst1 = [[1], [2], [4], [4], [5]]
    >>> X1 = nimble.data(lst1)
    >>> percentileNormalize(X1)
    <Matrix 5pt x 1ft
           0
       ┌──────
     0 │ 0.000
     1 │ 0.250
     2 │ 0.625
     3 │ 0.625
     4 │ 1.000
    >
    >>> lst2 = [[3], [2], [6]]
    >>> X2 = nimble.data(lst2)
    >>> norm1, norm2 = percentileNormalize(X1, X2)
    >>> norm2
    <Matrix 3pt x 1ft
           0
       ┌──────
     0 │ 0.438
     1 │ 0.250
     2 │ 1.000
    >
    """
    arr1 = values1.copy('numpyarray', outputAs1D=True).astype(float)
    percentiles = arr1.copy()

    # index locations of sorted values
    argsort = np.argsort(arr1)
    anyNan = any(map(np.isnan, arr1))
    if anyNan: # ignore index for nan
        epsilon = 1 / (len(arr1) - 2)
    else:
        epsilon = 1 / (len(arr1) - 1)
    for i, idx in enumerate(argsort):
        if not np.isnan(percentiles[idx]):
            percentiles[idx] = i * epsilon

    # adjust percentiles for duplicate values
    for v in np.unique(arr1):
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
    minimum = np.nanmin(sortedArr1)
    maximum = np.nanmax(sortedArr1)

    interpVal = functools.partial(_interpPercentile, minimum, maximum,
                                  sortedArr1, sortedPercs)
    interpVector = np.vectorize(interpVal)
    values2Percentiles = values2.copy()
    if len(values2.points) == 1:
        values2Percentiles.points.transform(interpVector, useLog=False)
    else:
        values2Percentiles.features.transform(interpVector, useLog=False)

    return values1Percentiles, values2Percentiles

def _interpPercentile(minimum, maximum, sortedValues, sortedPercentiles, val):
    if np.isnan(val):
        return val
    if val == minimum == maximum:
        return 0.5 # all values were equal
    if val <= minimum:
        return 0.0
    if val >= maximum:
        return 1.0
    idx = np.searchsorted(sortedValues, val) # binary search
    leftVal, rightVal = sortedValues[idx - 1:idx + 1]
    leftPerc, rightPerc = sortedPercentiles[idx - 1:idx + 1]
    if val == rightVal:
        return rightPerc
    fractionToNext = (val - leftVal) / (rightVal - leftVal)
    percDist = rightPerc - leftPerc
    return float(leftPerc + percDist * fractionToNext)
