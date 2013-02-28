"""
Class to handle the calculation of various statistics about a data set and converting
them into a readable version suitable for printing to a log file.  Some statistics
that are calculated, depending on the type of feature: sparsity (proportion of zero valued
entries), min, max, mean, st. dev., # of unique values (for non-real valued features), # of
missing values
"""
import math

from UML.processing.base_data import BaseData
from UML.processing.row_list_data import RowListData
from UML.logging.tableString import tableString


class DataSetAnalyzer(object):

	def __init__(self, dataSet=None):
		"""
		Store the dataSet associated with this object.  dataSet should be an object of
		a type that inherits from the BaseData class.
		"""
		if dataSet is not None:
			self.dataSet = dataSet

	def setData(self, dataSet):
		"""
		Setter for this object's dataSet field.
		"""
		self.dataSet = dataSet

	def featurewiseFunctionGenerator(self):
		"""
		TODO: fill out body of function, which should return a list of default 
		functions that should be applied to each feature
		"""
		functions = [self.min_, self.max_, self.mean_, self.standardDeviation, self.numUnique]
		return functions

	def produceFeatureWiseReport(self, supplementalFunctions=None, maxFeaturesToCover=50):
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
		functionsToApply = self.featurewiseFunctionGenerator()
		if supplementalFunctions is not None:
			functionsToApply.extend(supplementalFunctions)

		columnLabels = []
		for func in functionsToApply:
			label = func.__name__
			columnLabels.append(label)

		rowLabelsList = [None] * len(self.dataSet.featureNames)
		for featureName, index in self.dataSet.featureNames.iteritems():
			rowLabelsList[index] = featureName

		for rowLabel in rowLabelsList:
			print rowLabel

		featurewiseResults = RowListData(rowLabelsList)
		#print len(featurewiseResults.data)
		featurewiseResults.transpose()
		print tableString(featurewiseResults.data, False)
		for func in functionsToApply:
			oneFuncResults = self.dataSet.applyFunctionToEachFeature(func)
			#print len(oneFuncResults.data)
			oneFuncResults.transpose()
			#print len(oneFuncResults.data)
			oneFuncResults.renameFeatureName(0, func.__name__)
			oneFuncResults = oneFuncResults.toRowListData()
			featurewiseResults.appendFeatures(oneFuncResults)

		table = tableString(featurewiseResults.data, True, columnLabels)

		return table


	def isNumerical(self, vector):
		"""
		Returns true if the vector contains primitive numerical non-complex values, 
		returns false otherwise.  Assumes that all items in vector are of the same type.
		"""
		for item in vector:
			if isinstance(item, (int, float, long)):
				return True
			else:
				return False


	def min_(self, values):
		"""
		Given a 1D vector of values, find the minimum value.  If the values are
		not numerical, return None.
		"""
		if not self.isNumerical(values):
			return None

		currMin = float("inf")

		for value in values:
			if not math.isnan(value) and value < currMin:
				currMin = value

		if currMin != float("-inf"):
			return currMin
		else:
			return None

	def max_(self, values):
		"""
		Given a 1D vector of values, find the maximum value.  If the values are
		not numerical, return None.
		"""
		if not self.isNumerical(values):
			return None

		currMax = float("-inf")

		for value in values:
			if not math.isnan(value) and value < currMax:
				currMax = value

		if currMax != float("-inf"):
			return currMax
		else:
			return None

	def mean_(self, values):
		"""
		Given a 1D vector of values, find the minimum value.  If the values are
		not numerical, return None.
		"""
		if not self.isNumerical(values):
			return None

		count = 0
		runningSum = 0

		for value in values:
			if not math.isnan(value):
				runningSum += value
				count += 1

		if count > 0:
			return float(runningSum) / count

	def standardDeviation(self, values):
		"""
		Given a 1D vector of values, find the standard deviation.  If the values are
		not numerical, return None.
		"""
		if not self.isNumerical(values):
			return None

		count = 0
		runningSum = 0

		for value in values:
			if not math.isnan(value):
				runningSum += value
				count += 1

		if not count > 0:
			return 0
		else:
			mean = float(runningSum)/count

		squaredDifferenceTotal = 0
		for value in values:
			squaredDifferenceTotal += (mean - value)**2

		stDev = math.sqrt(squaredDifferenceTotal / count)
		return stDev

	def numUnique(self, values):
		"""
		Given a 1D vector of values, calculate the number of unique values.
		"""
		valueSet = set(values)
		return len(valueSet)
