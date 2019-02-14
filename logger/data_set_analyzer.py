"""
Module to handle the calculation of various statistics about a data set and converting
them into a readable version suitable for printing to a log file.  Some statistics
that are calculated, depending on the type of feature: sparsity (proportion of zero valued
entries), min, max, mean, st. dev., # of unique values (for non-real valued features), # of
missing values
"""
from __future__ import absolute_import
import math

import UML
from .tableString import tableString
from UML.exceptions import InvalidArgumentValueCombination
import six
from six.moves import range


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

    resultsTable = [None] * len(dataContainer.features)
    for index in range(len(dataContainer.features)):
        resultsTable[index] = dataContainer.features.getName(index)

    transposeRow(resultsTable)

    for func in funcsToApply:
        oneFuncResults = dataContainer.features.calculate(func)
        oneFuncResults.transpose()
        oneFuncResultsList = oneFuncResults.copyAs(format="python list")
        appendColumns(resultsTable, oneFuncResultsList)

    #add Function names as the first row in the results table
    resultsTable.insert(0, columnLabels)

    return resultsTable


def produceFeaturewiseReport(dataContainer, supplementalFunctions=None, maxFeaturesToCover=50,
                             displayDigits=2, useLog=None):
    """
    Produce a string formatted as a table.  The table contains iformation about each feature
    in the data set contained in dataContainer.

    Inputs
        dataContainer - object of a type that inherits from Base.

        supplementalFunctions - additional functions that should be applied to each feature in
            the data set.  Arg signature must expect one iterable feature vector, any other parameters
            must be optional.

        maxFeaturesToCover - # of features (columns) in the data set to compute statistics for and
        include in the result.  If there are more features in the data set, an arbitrary subset of size
        equal to maxFeaturesToCoverwill be chosen, the rest will be ignored.
    """
    #try to find out basic information about the data set: # of rows
    #and columns.  For numpy containers, this is obtained through call
    #to shape().  For List, can use len()
    shape = computeShape(dataContainer)

    functionsToApply = featurewiseFunctionGenerator()
    if supplementalFunctions is not None:
        functionsToApply.extend(supplementalFunctions)

    #If the data object is too big to print out info about each feature,
    #extract a subset of features from the data set and
    if shape[1] > maxFeaturesToCover:
        if maxFeaturesToCover % 2 == 0:
            leftIndicesToSelect = list(range(maxFeaturesToCover / 2))
            rightIndicesToSelect = list(range(shape[1] - (maxFeaturesToCover / 2), shape[1]))
        else:
            leftIndicesToSelect = list(range(math.floor(maxFeaturesToCover / 2)))
            rightIndicesToSelect = list(range(shape[1] - ((maxFeaturesToCover / 2) + 1), shape[1]))
        subsetIndices = []
        subsetIndices.extend(leftIndicesToSelect)
        subsetIndices.extend(rightIndicesToSelect)
        dataContainer = dataContainer.features.copy(subsetIndices)
        isSubset = True
    else:
        isSubset = False

    infoTable = produceFeaturewiseInfoTable(dataContainer, functionsToApply)

    if displayDigits is not None and isinstance(displayDigits, six.integer_types):
        displayDigits = "." + str(displayDigits) + "f"

    if isSubset:
        printableTable = tableString(infoTable, True, headers=infoTable[0], roundDigits=displayDigits,
                                     snipIndex=leftIndicesToSelect[-1])
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
        funcResults = dataContainer.features.calculate(func)
        funcResults.transpose()
        aggregateResults = funcResults.features.calculate(UML.calculate.mean).copyAs(format="python list")[0][0]
        resultsDict[func.__name__] = aggregateResults

    resultsDict['Values'] = shape[0] * shape[1]
    resultsDict['Points'] = shape[0]
    resultsDict['Features'] = shape[1]

    headers = []
    stats = []
    for header, value in six.iteritems(resultsDict):
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

    if displayDigits is not None and isinstance(displayDigits, six.integer_types):
        displayDigits = "." + str(displayDigits) + "f"

    return tableString(table, False, headers=table[0], roundDigits=displayDigits)

###############################################################################
#                                                                             #
#                           Utility functions                                 #
#                                                                             #
###############################################################################

def featurewiseFunctionGenerator():
    """
    Produce a list of functions suitable for being passed to Base's calculateForEachFeature
    function.  Includes: min(), max(), mean(), median(), standardDeviation(), numUniqueValues()
    """
    functions = [UML.calculate.minimum, UML.calculate.maximum, UML.calculate.mean,
                 UML.calculate.median, UML.calculate.standardDeviation,
                 UML.calculate.uniqueCount]
    return functions


def aggregateFunctionGenerator():
    """
    Produce a list of functions that can be used to produce aggregate statistics on an entire
    data set.  The functions will be applied through Base's calculateForEachFeature
    function, and their results averaged across all features.  Includes a function to calculate
    the proportion of entries that are equal to zero and the proportion of entries that are missing
    (i.e. are None or NaN).
    """
    functions = [UML.calculate.proportionZero, UML.calculate.proportionMissing]
    return functions


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
        If they do not have the same number of rows, an InvalidArgumentValueCombination
        is raised.
    """
    if len(appendTo) != len(appendFrom):
        raise InvalidArgumentValueCombination("Can't merge two matrices with different numbers of rows: " +
                                str(len(appendTo)) + " != " + str(len(appendFrom)))

    for i in range(len(appendTo)):
        appendFromRow = appendFrom[i]
        for j in range(len(appendFromRow)):
            appendTo[i].append(appendFromRow[j])

    return


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
