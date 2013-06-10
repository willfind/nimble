"""
Module to handle the calculation of various statistics about a data set and converting
them into a readable version suitable for printing to a log file.  Some statistics
that are calculated, depending on the type of feature: sparsity (proportion of zero valued
entries), min, max, mean, st. dev., # of unique values (for non-real valued features), # of
missing values
"""
import math
import numpy as np
from nose.tools import assert_equal
from nose.tools import assert_almost_equal

import UML
from UML.uml_logging.tableString import tableString
from UML.utility import ArgumentException


def produceFeaturewiseInfoTable(dataContainer, funcsToApply):
    """
    Produce a report with summary information about each feature in the
    data set found in dataContainer.  Returns the report as a string containing a 2-dimensional
    table, with each row representing a feature and each column representing
    a piece of information about the feature.  If the number of features in
    the data set is greater than maxFeaturesToCover, only a subset (equal in size
    to maxFeaturesToCover) of features will be included in the report.
    Input:
        supplementalFunctions:  list of supplemental functions to be applied to each feature,
        which must accept a 1D vector and return a scalar value.  These will be applied
        in addition to the default functions (min, max, mean, st. dev, # unique values,
        # missing values, # zero values, etc.).

        maxFeaturesToCover: if the number of features exceeds maxFeaturesToCover, then
        summary information will only be calculated and reported for a subset of features,
        equal in size to maxFeaturesToCover.

    Output:
        A table with each row containing summary information about one
        of the features in the data set.  Each column represents the results of one of
        the functions applied to each feature.
    """
    columnLabels = ['featureName']
    for func in funcsToApply:
        label = func.__name__.rstrip('_')
        columnLabels.append(label)

    resultsTable = [None] * len(dataContainer.featureNames)
    for featureName, index in dataContainer.featureNames.iteritems():
        resultsTable[index] = featureName

    transposeRow(resultsTable)

    for func in funcsToApply:
        oneFuncResults = dataContainer.applyFunctionToEachFeature(func)
        oneFuncResults.transpose()
        oneFuncResultsList = oneFuncResults.toListOfLists()
        appendColumns(resultsTable, oneFuncResultsList)

    #add Function names as the first row in the results table
    resultsTable.insert(0, columnLabels)

    return resultsTable


def produceFeaturewiseReport(dataContainer, supplementalFunctions=None, maxFeaturesToCover=50, displayDigits=2):
    """
    Produce a string formatted as a table.  The table contains iformation about each feature
    in the data set contained in dataContainer.

    Inputs
        dataContainer - object of a type that inherits from BaseData.
        
        supplementalFunctions - additional functions that should be applied to each feature in 
            the data set.  Arg signature must expect one iterable feature vector, any other parameters
            must be optional.

        maxFeaturesToCover - # of features (columns) in the data set to compute statistics for and
        include in the result.  If there are more features in the data set, an arbitrary subset of size
        equal to maxFeaturesToCoverwill be chosen, the rest will be ignored.
    """
    #try to find out basic information about the data set: # of rows
    #and columns.  For numpy containers, this is obtained through call
    #to shape().  For RowListData, can use len()
    shape = computeShape(dataContainer)

    functionsToApply = featurewiseFunctionGenerator()
    if supplementalFunctions is not None:
        functionsToApply.extend(supplementalFunctions)

    #If the data object is too big to print out info about each feature,
    #extract a subset of features from the data set and 
    if shape[0] > maxFeaturesToCover:
        if maxFeaturesToCover % 2 == 0:
            leftIndicesToSelect = range(maxFeaturesToCover / 2)
            rightIndicesToSelect = range(shape[0] - (maxFeaturesToCover / 2), shape[0])
        else:
            leftIndicesToSelect = range(math.floor(maxFeaturesToCover / 2))
            rightIndicesToSelect = range(shape[0] - ((maxFeaturesToCover / 2) + 1), shape[0])
        dataContainer = dataContainer.copyFeatures(leftIndicesToSelect.append(rightIndicesToSelect))
        isSubset = True
    else:
        isSubset = False

    infoTable = produceFeaturewiseInfoTable(dataContainer, functionsToApply)

    if displayDigits is not None and isinstance(displayDigits, (int, long)):
        displayDigits = "." + str(displayDigits) + "f"

    if isSubset:
        printableTable = tableString(infoTable, True, headers=infoTable[0], roundDigits=displayDigits, snipIndex=leftIndicesToSelect[-1])
    else:
        printableTable = tableString(infoTable, True, headers=infoTable[0], roundDigits=displayDigits)

    return printableTable


def produceAggregateTable(dataContainer):
    """
    Calculate various aggregate statistics about the data set held in dataContainer and return
    them as a table (2d list), with descriptors in the first row and numerical results of those descriptors
    in the second row of the table.

    Statistics gathered:  proportion of zero values, proportion of missing values, number of Points
    in the data set, number of features in the data set.
    """
    shape = computeShape(dataContainer)
    funcs = aggregateFunctionGenerator()
    resultsDict = {}
    for func in funcs:
        funcResults = dataContainer.applyFunctionToEachFeature(func)
        funcResults.transpose()
        aggregateResults = funcResults.applyFunctionToEachFeature(mean_).toListOfLists()[0][0]
        resultsDict[func.__name__] = aggregateResults

    resultsDict['Points'] = shape[0] * shape[1]
    resultsDict['Features'] = shape[1]

    headers = []
    stats = []
    for header, value in resultsDict.iteritems():
        headers.append(header)
        stats.append(value)

    return [headers, stats]


def produceAggregateReport(dataContainer, displayDigits):
    """
    Calculate various aggregate statistics about the data set held in dataContainer and return
    them as a string containing a table, with descriptors in the first row and numerical results 
    of those descriptors in the second row of the table.

    Statistics gathered:  proportion of zero values, proportion of missing values, number of Points
    in the data set, number of features in the data set.
    """
    table = produceAggregateTable(dataContainer)

    if displayDigits is not None and isinstance(displayDigits, (int, long)):
        displayDigits = "." + str(displayDigits) + "f"

    return tableString(table, False, headers=table[0], roundDigits=displayDigits)



###############################################################################
#                                                                             #
#                           Statistical functions                             #
#                                                                             #
###############################################################################


def proportionMissing(values):
    """
    Calculate proportion of entries in 'values' iterator that are missing (defined
    as being None or NaN).
    """
    numMissing = 0
    numTotal = len(values)
    for value in values:
        if isMissing(value):
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


def min_(values):
    """
    Given a 1D vector of values, find the minimum value.  If the values are
    not numerical, return None.
    """
    if not isNumericalFeatureGuesser(values):
        return None

    currMin = float("inf")
    nonZeroValues = values.nonZeroIterator()
    count = 0

    for value in nonZeroValues:
        count += 1
        if isNumericalPoint(value) and value < currMin:
            currMin = value

    if not math.isinf(currMin):
        if currMin > 0 and len(values) > count:
            return 0
        else: return currMin
    else:
        return None


def max_(values):
    """
    Given a 1D vector of values, find the maximum value.  If the values are
    not numerical, return None.
    """
    if not isNumericalFeatureGuesser(values):
        return None

    currMax = float("-inf")
    nonZeroValues = values.nonZeroIterator()
    count = 0

    for value in nonZeroValues:
        count += 1
        if isNumericalPoint(value) and value > currMax:
            currMax = value

    if not math.isinf(currMax):
        if currMax < 0 and count < len(values):
            return 0
        else: return currMax
    else:
        return None


def mean_(values):
    """
    Given a 1D vector of values, find the minimum value.  If the values are
    not numerical, return None.
    """
    if not isNumericalFeatureGuesser(values):
        return None

    numericalCount = 0
    nonZeroCount = 0
    runningSum = 0
    totalCount = len(values)
    nonZeroValues = values.nonZeroIterator()

    for value in nonZeroValues:
        nonZeroCount +=1
        if isNumericalPoint(value):
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


def median_(values):
    """
    Given a 1D vector of values, find the minimum value.  If the values are
    not numerical, return None.
    """
    if not isNumericalFeatureGuesser(values):
        return None

    #Filter out None/NaN values from list of values
    sortedValues = filter(lambda x: not (x == None or math.isnan(x)), values)

    sortedValues = sorted(sortedValues)

    numValues = len(sortedValues)

    if numValues % 2 == 0:
        median = (float(sortedValues[(numValues/2) - 1]) + float(sortedValues[numValues/2])) / float(2)
    else:
        median = float(sortedValues[int(math.floor(numValues/2))])

    return median


def standardDeviation(values):
    """
    Given a 1D vector of values, find the standard deviation.  If the values are
    not numerical, return None.
    """
    if not isNumericalFeatureGuesser(values):
        return None

    #Filter out None/NaN values from list of values
    mean = mean_(values)
    nonZeroCount = 0
    numericalCount = 0
    nonZeroValues = values.nonZeroIterator()

    squaredDifferenceTotal = 0
    for value in nonZeroValues:
        nonZeroCount += 1
        if isNumericalPoint(value):
            numericalCount += 1
            squaredDifferenceTotal += (mean - value)**2

    if nonZeroCount < len(values):
        numZeros = len(values) - nonZeroCount
        squaredDifferenceTotal += numZeros * mean**2
        numericalCount += numZeros

    stDev = math.sqrt(squaredDifferenceTotal / float(numericalCount))
    return stDev


def numUnique(values):
    """
    Given a 1D vector of values, calculate the number of unique values.
    """
    values = filter(lambda x: not (x == None or math.isnan(x)), values)
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


###############################################################################
#                                                                             #
#                           Utility functions                                 #
#                                                                             #
###############################################################################

def featurewiseFunctionGenerator():
    """
    Produce a list of functions suitable for being passed to BaseData's applyFunctionToEachFeature
    function.  Includes: min(), max(), mean(), median(), standardDeviation(), numUniqueValues()
    """
    functions = [min_, max_, mean_, median_, standardDeviation, numUnique]
    return functions


def aggregateFunctionGenerator():
    """
    Produce a list of functions that can be used to produce aggregate statistics on an entire
    data set.  The functions will be applied through BaseData's applyFunctionToEachFeature
    function, and their results averaged across all features.  Includes a function to calculate
    the proportion of entries that are equal to zero and the proportion of entries that are missing
    (i.e. are None or NaN).
    """
    functions = [proportionZero, proportionMissing]
    return functions


def isMissing(point):
    """
    Determine if a point is missing or not.  If the point is None or NaN, return True.
    Else return False.
    """
    if point is None or math.isnan(point):
        return True
    else: return False


def transposeRow(row):
    """
        Given a row (1d list), convert it into a column (list of lists, each
        with one element of the original row.  Alters row directly, rather than
        making a copy and returning the copy.
        Example: [1, 2, 3, 4] -> [[1], [2], [3], [4]]
        TODO: make more general (just transpose, rather than transposeRow)
    """
    for i in range(len(row)):
        row[i] = [row[i]]

    return


def appendColumns(appendTo, appendFrom):
    """
        Append the columns of one 2D matrix (lists of lists) into another.  
        They must have the same number of rows, but can have different numbers 
        of columns.  I.e. len(appendTo) == len(appendFrom), but 
        len(appendTo[0]) == len(appendFrom[0]) does not need to be true.
        If they do not have the same number of rows, an ArgumentException is
        raised.
    """
    if len(appendTo) != len(appendFrom):
        raise ArgumentException("Can't merge two matrices with different numbers of rows: " + 
            str(len(appendTo)) + " != " + str(len(appendFrom)))

    for i in xrange(len(appendTo)):
        appendFromRow = appendFrom[i]
        for j in xrange(len(appendFromRow)):
            appendTo[i].append(appendFromRow[j])

    return


def isNumericalFeatureGuesser(featureVector):
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


def isNumericalPoint(point):
    """
    Check to see if a point is a valid number that can be used in numerical calculations.
    If point is of type float, long, or int, and not None or NaN, return True.  Otherwise
    return False.
    """
    if isinstance(point, (int, float, long)) and not math.isnan(point):
        return True
    else:
        return False


def computeShape(dataContainer):
    """
    Compute number of rows and columns in the dataSet contained within dataContainer.  Return
    as a tuple: (numRows, numColumns)
    """
    try:
        shape = dataContainer.data.shape
    except AttributeError:
        shape = (len(dataContainer.data), len(dataContainer.data[0]))

    return shape


###############################################################################
#                                                                             #
#                                 Unit tests                                  #
#                                                                             #
###############################################################################

def testDense():
    """
        Test the functionality of calculating statistical/informational 
        funcs on a dense BaseData object using the produceInfoTable.
    """
    data1 = np.array([[1, 2, 3, 1], [3, 3, 1, 5], [1, 1, 5, 2]])
    names1 = ['var1', 'var2', 'var3', 'var4']

    trainObj = data('DenseMatrixData', data1, names1)
    funcs = featurewiseFunctionGenerator()
    rawTable = produceFeaturewiseInfoTable(trainObj, funcs)
    funcNames = rawTable[0]
    for i in range(len(funcNames)):
        funcName = funcNames[i]
        if funcName == "mean_" or funcName == "mean":
            assert_almost_equal(rawTable[1][i], 1.6667, 3)
            assert_almost_equal(rawTable[2][i], 2.000, 3)
            assert_almost_equal(rawTable[3][i], 3.000, 3)
            assert_almost_equal(rawTable[4][i], 2.6667, 3)
        elif funcName == "min_" or funcName == "min":
            assert_equal(rawTable[1][i], 1)
            assert_equal(rawTable[2][i], 1)
            assert_equal(rawTable[3][i], 1)
            assert_equal(rawTable[4][i], 1)
        elif funcName == "max_" or funcName == "max":
            assert_equal(rawTable[1][i], 3)
            assert_equal(rawTable[2][i], 3)
            assert_equal(rawTable[3][i], 5)
            assert_equal(rawTable[4][i], 5)
        elif funcName == "numUnique":
            assert_equal(rawTable[1][i], 2)
            assert_equal(rawTable[2][i], 3)
            assert_equal(rawTable[3][i], 3)
            assert_equal(rawTable[4][i], 3)
        elif funcName == "standardDeviation":
            assert_almost_equal(rawTable[1][i], 0.9428, 3)
            assert_almost_equal(rawTable[2][i], 0.8165, 3)
            assert_almost_equal(rawTable[3][i], 1.633, 3)
            assert_almost_equal(rawTable[4][i], 1.6997, 3)
        elif funcName == "median_" or funcName == "median":
            assert_equal(rawTable[1][i], 1.0)
            assert_equal(rawTable[2][i], 2.0)
            assert_equal(rawTable[3][i], 3.0)
            assert_equal(rawTable[4][i], 2.0)


def testSparse():
    """
        Test the functionality of calculating statistical/informational 
        funcs on a dense BaseData object using the produceInfoTable.

        Testing matrix is 6 x 6, with 11 non-zero values, 1 None/Missing value,
        and 24 zero values.
    """
    row = np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5])
    col = np.array([0, 4, 2, 3, 1, 3, 4, 0, 1, 3, 4, 5])
    vals = np.array([1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1])

    testObj = data('coo', (vals, (row, col)))
    funcs = featurewiseFunctionGenerator()
    rawTable = produceFeaturewiseInfoTable(testObj, funcs)

    funcNames = rawTable[0]
    for i in range(len(funcNames)):
        funcName = funcNames[i]
        if funcName == "mean_" or funcName == "mean":
            assert_almost_equal(rawTable[1][i], 0.3333, 3)
            assert_almost_equal(rawTable[2][i], 0.3333, 3)
            assert_almost_equal(rawTable[3][i], 0.1667, 3)
            assert_almost_equal(rawTable[4][i], 0.4000, 3)
            assert_almost_equal(rawTable[5][i], 0.5000, 3)
            assert_almost_equal(rawTable[6][i], 0.1667, 3)
        elif funcName == "min_" or funcName == "min":
            assert_equal(rawTable[1][i], 0)
            assert_equal(rawTable[2][i], 0)
            assert_equal(rawTable[3][i], 0)
            assert_equal(rawTable[4][i], 0)
            assert_equal(rawTable[5][i], 0)
            assert_equal(rawTable[6][i], 0)
        elif funcName == "max_" or funcName == "max":
            assert_equal(rawTable[1][i], 1)
            assert_equal(rawTable[2][i], 1)
            assert_equal(rawTable[3][i], 1)
            assert_equal(rawTable[4][i], 1)
            assert_equal(rawTable[5][i], 1)
            assert_equal(rawTable[6][i], 1)
        elif funcName == "numUnique":
            assert_equal(rawTable[1][i], 2)
            assert_equal(rawTable[2][i], 2)
            assert_equal(rawTable[3][i], 2)
            assert_equal(rawTable[4][i], 2)
            assert_equal(rawTable[5][i], 2)
            assert_equal(rawTable[6][i], 2)
        elif funcName == "standardDeviation":
            assert_almost_equal(rawTable[1][i], 0.4714, 3)
            assert_almost_equal(rawTable[2][i], 0.4714, 3)
            assert_almost_equal(rawTable[3][i], 0.3727, 3)
            assert_almost_equal(rawTable[4][i], 0.4899, 3)
            assert_almost_equal(rawTable[5][i], 0.500, 3)
            assert_almost_equal(rawTable[6][i], 0.3727, 3)
        elif funcName == "median_" or funcName == "median":
            assert_equal(rawTable[1][i], 0)
            assert_equal(rawTable[2][i], 0)
            assert_equal(rawTable[3][i], 0)
            assert_equal(rawTable[4][i], 0)
            assert_equal(rawTable[5][i], 0.5)
            assert_equal(rawTable[6][i], 0)


def testRowList():
    data1 = np.array([[1, 2, 3, 1], [3, 3, 1, 5], [1, 1, 5, 2]])
    names1 = ['var1', 'var2', 'var3', 'var4']

    trainObj = data('RowListData', data1, names1)
    funcs = featurewiseFunctionGenerator()
    rawTable = produceFeaturewiseInfoTable(trainObj, funcs)
    funcNames = rawTable[0]
    for i in range(len(funcNames)):
        funcName = funcNames[i]
        if funcName == "mean_" or funcName == "mean":
            assert_almost_equal(rawTable[1][i], 1.6667, 3)
            assert_almost_equal(rawTable[2][i], 2.000, 3)
            assert_almost_equal(rawTable[3][i], 3.000, 3)
            assert_almost_equal(rawTable[4][i], 2.6667, 3)
        elif funcName == "min_" or funcName == "min":
            assert_equal(rawTable[1][i], 1)
            assert_equal(rawTable[2][i], 1)
            assert_equal(rawTable[3][i], 1)
            assert_equal(rawTable[4][i], 1)
        elif funcName == "max_" or funcName == "max":
            assert_equal(rawTable[1][i], 3)
            assert_equal(rawTable[2][i], 3)
            assert_equal(rawTable[3][i], 5)
            assert_equal(rawTable[4][i], 5)
        elif funcName == "numUnique":
            assert_equal(rawTable[1][i], 2)
            assert_equal(rawTable[2][i], 3)
            assert_equal(rawTable[3][i], 3)
            assert_equal(rawTable[4][i], 3)
        elif funcName == "standardDeviation":
            assert_almost_equal(rawTable[1][i], 0.9428, 3)
            assert_almost_equal(rawTable[2][i], 0.8165, 3)
            assert_almost_equal(rawTable[3][i], 1.633, 3)
            assert_almost_equal(rawTable[4][i], 1.6997, 3)
        elif funcName == "median_" or funcName == "median":
            assert_equal(rawTable[1][i], 1.0)
            assert_equal(rawTable[2][i], 2.0)
            assert_equal(rawTable[3][i], 3.0)
            assert_equal(rawTable[4][i], 2.0)

def testAppendColumns():
    """
        Unit test for appendColumns function in data_set_analyzer
    """
    table1 = [[1, 2, 3], [7, 8, 9]]
    table2 = [[4, 5, 6], [10, 11, 12]]

    table3 = [[1], [2], [3], [4]]
    table4 = [[5], [6], [7], [8]]

    table5 = [["one"], ["two"], ["three"], ["four"]]
    table6 = [[1], [2], [3], [4]]

    appendColumns(table1, table2)
    appendColumns(table3, table4)
    appendColumns(table5, table6)

    assert len(table1) == 2
    assert len(table1[0]) == 6
    assert table1[0][2] == 3
    assert table1[0][3] == 4
    assert table1[1][2] == 9
    assert table1[1][3] == 10

    assert len(table3) == 4
    assert len(table3[0]) == 2
    assert table3[0][0] == 1
    assert table3[0][1] == 5
    assert table3[1][0] == 2
    assert table3[1][1] == 6
    assert table3[2][0] == 3
    assert table3[2][1] == 7
    assert table3[3][0] == 4
    assert table3[3][1] == 8

    assert len(table5) == 4
    assert len(table5[0]) == 2
    assert table5[0][0] == 'one'
    assert table5[0][1] == 1
    assert table5[1][0] == 'two'
    assert table5[1][1] == 2
    assert table5[2][0] == 'three'
    assert table5[2][1] == 3
    assert table5[3][0] == 'four'
    assert table5[3][1] == 4

def testProduceAggregateTable():
    """
    TODO: add docstring
    """
    data1 = np.array([[1, 2, 3, 1], [3, 3, 1, 5], [1, 1, 5, 2]])
    names1 = ['var1', 'var2', 'var3', 'var4']

    trainObj = data('RowListData', data1, names1)
    rawTable = produceAggregateTable(trainObj)

    for i in range(len(rawTable[0])):
        funcName = rawTable[0][i]
        if funcName == "proportionZero":
            assert rawTable[1][i] == 0.0
        elif funcName == "proportionMissing":
            assert rawTable[1][i] == 0.0
        elif funcName == "Points":
            assert rawTable[1][i] == 12
        elif funcName == "Features":
            assert rawTable[1][i] == 4


def testStDev():
    testRowList = data('rld', np.array([[1], [1], [3], [4], [2], [6], [12], [0]]), ['nums'])
    stDevContainer = testRowList.applyFunctionToEachFeature(standardDeviation)
    stDev = stDevContainer.toListOfLists()[0][0]
    assert_almost_equal(stDev, 3.6379, 3)


