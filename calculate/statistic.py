import math


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
        else: pass

    if numTotal > 0:
        return float(numMissing) / float(numTotal)
    else: return 0.0


def proportionZero(values):
    """
    Calculate proportion of entries in 'values' iterator that are equal to zero.
    """
    totalNum = len(values)
    nonZeroCount = 0
    nonZeroItr = values.nonZeroIterator()
    for value in nonZeroItr:
        nonZeroCount += 1

    if totalNum > 0:
        return float(totalNum - nonZeroCount) / float(totalNum)
    else: return 0.0


def minimum(values):
    """
    Given a 1D vector of values, find the minimum value.  If the values are
    not numerical, return None.
    """
    if not _isNumericalFeatureGuesser(values):
        return None

    currMin = float("inf")
    nonZeroValues = values.nonZeroIterator()
    count = 0

    for value in nonZeroValues:
        count += 1
        if _isNumericalPoint(value) and value < currMin:
            currMin = value

    if not math.isinf(currMin):
        if currMin > 0 and len(values) > count:
            return 0
        else: return currMin
    else:
        return None


def maximum(values):
    """
    Given a 1D vector of values, find the maximum value.  If the values are
    not numerical, return None.
    """
    if not _isNumericalFeatureGuesser(values):
        return None

    currMax = float("-inf")
    nonZeroValues = values.nonZeroIterator()
    count = 0

    for value in nonZeroValues:
        count += 1
        if _isNumericalPoint(value) and value > currMax:
            currMax = value

    if not math.isinf(currMax):
        if currMax < 0 and count < len(values):
            return 0
        else: return currMax
    else:
        return None


def mean(values):
    """
    Given a 1D vector of values, find the minimum value.  If the values are
    not numerical, return None.
    """
    if not _isNumericalFeatureGuesser(values):
        return None

    numericalCount = 0
    nonZeroCount = 0
    runningSum = 0
    totalCount = len(values)
    nonZeroValues = values.nonZeroIterator()

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
    Given a 1D vector of values, find the minimum value.  If the values are
    not numerical, return None.
    """
    if not _isNumericalFeatureGuesser(values):
        return None

    #Filter out None/NaN values from list of values
    sortedValues = filter(lambda x: not (x is None or math.isnan(x)), values)

    sortedValues = sorted(sortedValues)

    numValues = len(sortedValues)

    if numValues % 2 == 0:
        median = (float(sortedValues[(numValues/2) - 1]) + float(sortedValues[numValues/2])) / float(2)
    else:
        median = float(sortedValues[int(math.floor(numValues/2))])

    return median


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
    nonZeroValues = values.nonZeroIterator()

    squaredDifferenceTotal = 0
    for value in nonZeroValues:
        nonZeroCount += 1
        if _isNumericalPoint(value):
            numericalCount += 1
            squaredDifferenceTotal += (meanRet - value)**2

    if nonZeroCount < len(values):
        numZeros = len(values) - nonZeroCount
        squaredDifferenceTotal += numZeros * meanRet**2
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
    values = filter(lambda x: not (x is None or math.isnan(x)), values)
    valueSet = set(values)
    return len(valueSet)


def featureType(values):
    """
        Return the type of data: string, int, float
        TODO: add numpy type checking
    """
    for value in values:
        if isinstance(value, str):
            return "string"
        elif isinstance(value, (int, long)):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, complex):
            return "complex"
        else:
            pass

    return "Unknown"

def _isMissing(point):
    """
    Determine if a point is missing or not.  If the point is None or NaN, return True.
    Else return False.
    """
    if point is None or math.isnan(point):
        return True
    else: return False

def _isNumericalFeatureGuesser(featureVector):
    """
    Returns true if the vector contains primitive numerical non-complex values, 
    returns false otherwise.  Assumes that all items in vector are of the same type.
    """
    for item in featureVector:
        if isinstance(item, (int, float, long)):
            return True
        elif item is None or math.isnan(item):
            pass
        else:
            return False

    return True


def _isNumericalPoint(point):
    """
    Check to see if a point is a valid number that can be used in numerical calculations.
    If point is of type float, long, or int, and not None or NaN, return True.  Otherwise
    return False.
    """
    if isinstance(point, (int, float, long)) and not math.isnan(point):
        return True
    else:
        return False
