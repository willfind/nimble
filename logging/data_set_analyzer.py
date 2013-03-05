"""
Class to handle the calculation of various statistics about a data set and converting
them into a readable version suitable for printing to a log file.  Some statistics
that are calculated, depending on the type of feature: sparsity (proportion of zero valued
entries), min, max, mean, st. dev., # of unique values (for non-real valued features), # of
missing values
"""
import math
import numpy as np

from UML import data
from UML.processing.base_data import BaseData
from UML.processing.row_list_data import RowListData
from UML.logging.tableString import tableString
from UML.utility import ArgumentException



def featurewiseFunctionGenerator(self):
	"""
	TODO: fill out docstring
	"""
	functions = [min_, max_, mean_, median_, standardDeviation, numUnique, featureType, numValues]
	return functions

def produceFeatureWiseReport(dataContainer, supplementalFunctions=None, maxFeaturesToCover=50):
	"""
	Produce a report with summary information about each feature in the
	data set.  Returns the report as a string containing a 2-dimensional 
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
		A string containing a table with each row containing summary information about one
		of the features in the data set.  Each column represents the results of one of
		the functions applied to each feature.
	"""
	functionsToApply = featurewiseFunctionGenerator()
	if supplementalFunctions is not None:
		functionsToApply.extend(supplementalFunctions)

	columnLabels = ['featureName']
	for func in functionsToApply:
		label = func.__name__
		columnLabels.append(label)

    #try to find out basic information about the data set: # of rows
    #and columns.  For numpy containers, this is obtained through call
    #to shape().  For RowListData, can use len()
    try:
        numRows = dataContainer.data.shape[0]
        numColumns = dataContainer.data.shape[1]
    except AttributeError: 
        numRows = len(dataContainer.data[0])
        numColumns = len(dataContainer.data)

    #If the data object is too big to print out info about each feature,
    #get a 
    if numColumns > maxFeaturesToCover:
        leftFeatureIndices = range(maxFeaturesToCover/2)
        leftSlice = dataContainer.copyFeatures(leftFeatureIndices)

        rightFeatureIndices = 

	rowLabelsList = [None] * len(dataSet.featureNames)
	for featureName, index in dataSet.featureNames.iteritems():
		rowLabelsList[index] = [featureName]

	for func in functionsToApply:
		oneFuncResults = dataSet.applyFunctionToEachFeature(func)
		oneFuncResults.transpose()
		oneFuncResultsList = oneFuncResults.toListOfLists()
		appendColumns(rowLabelsList, oneFuncResultsList)

    #In the case that this is a sparse matrix, we need to adjust the statistics
    #to account for missing zeros
    if rowLabelsList[1][-1] != numRows:
        for row in rowLabelsList[1:]:
            row = adjustStats(row, rowLabelsList[0], numRows)

	return table

def convertToPrintableTable(table, headers):
    """
        TODO: add docstring
    """
    return tableString(table, True, headers)

def adjustStats(rawStats, funcNames, totalNumRows):
    """
        TODO: add docstring
    """
    rawNumRows = rawStats[-1]

    #find value of 'mean' for this feature
    for i in range(rawStats):
        if funcNames[i] == 'mean_':
            mean = rawStats[i]
        else:
            pass

    for i in range(rawStats):
        rawStat = rawStats[i]
        funcName = funcNames[i]
        if funcName == "min_":
            if rawStat > 0:
                rawStats[i] = 0
            else:
                rawStats[i] = rawStat
        elif funcName == "max_":
            if rawStat < 0:
                rawStats[i] = 0
            else:
                rawStats[i] = rawStat
        elif funcName == "median_":

        elif funcName == "mean_":
            rawStats[i] = = float(rawStat * rawNumRows) / float(totalNumRows)
        elif funcName == "standardDeviation":
            rawStats[i] = math.sqrt(rawStats[i]**2 + (totalNumRows - rawNumRows)*(mean**2))
        elif funcName == "numUnique":
            rawStats[i] = rawStat + 1
        elif funcName == 'featureType':
            rawStats[i] = rawStat
        else:
            rawStats[i] = rawStat

    return rawStats


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
			len(appendTo) + " != " + len(appendFrom))

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


def min_(values):
	"""
	Given a 1D vector of values, find the minimum value.  If the values are
	not numerical, return None.
	"""
	if not isNumericalFeatureGuesser(values):
		return None

	currMin = float("inf")
	count = 0

	for value in values:
		count += 1
		if isNumericalPoint(value) and value < currMin:
			currMin = value

	if not math.isinf(currMin):
		return currMin
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

	count = 0

	for value in values:
		count += 1
		if isNumericalPoint(value) and value > currMax:
			currMax = value

	if not math.isinf(currMax):
		return currMax
	else:
		return None

def mean_(values):
	"""
	Given a 1D vector of values, find the minimum value.  If the values are
	not numerical, return None.
	"""
	if not isNumericalFeatureGuesser(values):
		return None

	totalCount = 0
	count = 0
	runningSum = 0

	for value in values:
		totalCount += 1
		if isNumericalPoint(value):
			runningSum += value
			count += 1

	if count > 0:
		return float(runningSum) / count, totalCount
	else:
		return None

def median_(values):
	"""
	Given a 1D vector of values, find the minimum value.  If the values are
	not numerical, return None.
	TODO: handle nan/None values
	"""
	if not isNumericalFeatureGuesser(values):
		return None

	count = 0
	for value in values:
		count += 1

	#Filter out None/NaN values from list of values
	sortedValues = filter(lambda x: not (x == None or math.isnan(x)), values)

	sortedValues = sorted(values)

	numValues = len(sortedValues)

	if numValues % 2 == 0:
		median = float(sortedValues[numValues/2])
	else:
		median = (sortedValues[(numValues + 1) / 2] + sortedValues[(numValues - 1) / 2]) / float(2)

	return median

def standardDeviation(values):
	"""
	Given a 1D vector of values, find the standard deviation.  If the values are
	not numerical, return None.
	"""
	if not isNumericalFeatureGuesser(values):
		return None

	#Filter out None/NaN values from list of values
	mean = mean_(values)[0]
	count = 0

	squaredDifferenceTotal = 0
	for value in values:
		count += 1
		if isNumericalPoint(value):
			squaredDifferenceTotal += (mean - value)**2

	stDev = math.sqrt(squaredDifferenceTotal / float(count))
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
        elif isinstance(value, (int, long))
            return "int"
        elif isinstance(value, (float)):
            return "float"
        elif isinstance(value, complex):
            return "complex"
        else:
            pass

    return "Unknown"


def numValues(values):
    """
        Compute the number of values in the feature, whether they are strings,
        numerical, None, or NaN.
    """
    count = 0
    for item in values:
        count += 1

    return count


def testDense():
    data1 = np.array([1, 2, 0, 1], [3, 3, 1, 0], [0, 0, 5, 2])
    names1 = ['var1', 'var2', 'var3', 'var4']

    trainObj = data('DenseMatrixData', data1, names1)
    rawTable = produceFeatureWiseReport(trainObj)
    funcNames = rawTable[0]
    funcIndex = 0
    for 


def testSparse():
    pass

def testRowList():
    pass
