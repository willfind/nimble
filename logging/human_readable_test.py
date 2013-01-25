# PEP 366 'boilerplate', plus the necessary import of the top level package
if __name__ == "__main__" and __package__ is None:
	import sys
	# add UML parent directory to sys.path
	sys.path.append(sys.path[0].rsplit('/',2)[0])
	import UML
	import UML.logging
	__package__ = "UML.logging"

import datetime
import numpy
import inspect

from tableString import *
from logger import Logger
from ..processing.coo_sparse_data import CooSparseData

class HumanReadableRunLog(Logger):

	def __init__(self, logFileName=None):
		super(HumanReadableRunLog, self).__init__(logFileName)


	def logRun(self, trainData, testData, function, metrics, extraInfo=None):
		"""
			Convert a set of objects representing one run (data used for training, data used for
			testing, function representing a unique classifier {algorithm, parameters}, error metrics,
			and any additional info) into a list.  This list can be appended to a second list, to create
			a 2-dimensional table that can be passed to tableString().  The results of tableString can
			then be printed to the log.

			Information contained, by default, in the resulting list:
				pass)
				# of training data points
				# of testing data points
				# of features in training data
				# of features in testing data
				Function defining the classifer (algorithm, parameters, etc.)
				Error metrics computed based on predictions of classifier: name/function and
				numerical result)

				Any additional information contained in extraInfo dict (optional).
		"""
		#if the log file is not available, try to create it
		if not self.isAvailable:
			self.setup()

		tableHeaders = []
		tableRow = []

		tableRow.append(str(datetime.datetime.now()))

		#add # of training points, # of of features to output list
		if trainData.data is not None:
			tableHeaders.append("# training instances")
			tableHeaders.append("# training features")
			tableRow.append(str(trainData.data.shape[0]))
			tableRow.append(str(trainData.data.shape[1]))
		else:
			tableHeaders.append("")
			tableHeaders.append("")
			tableRow.append("")
			tableRow.append("")
		
		#add # of testing points to output list
		if testData.data is not None:
			tableHeaders.append("# testing instances")
			tableHeaders.append("# testing features")
			tableRow.append(str(testData.data.shape[0]))
			tableRow.append(str(testData.data.shape[1]))
		else:
			tableHeaders.append("")
			tableHeaders.append("")
			tableRow.append("")
			tableRow.append("")


		#Write performance metrics
		#if there is one performance metric, add to the table.
		#Otherwise, put all metrics into a separate section.
		multipleMetrics = True
		if len(metrics) == 1:
			multipleMetrics = False
			for metric, result in metrics:
				tableHeaders.append("Error Rate")
				tableRow.append(str(result))

		#Print table w/basic info to the log
		basicTable = [tableHeaders, tableRow]
		basicTableStr = tableString(basicTable, True, None, roundDigits=4)
		self.logMessage(basicTableStr)

		#Print out the name/function text of the error metric being used (if there
		#is only one), or the rate & name/function text if more than one is being
		#used
		for metric, result in metrics:
			self.logMessage("METRIC: ")

			if inspect.isfunction(metric):
				metricFuncLines = inspect.getsourcelines(metric)
				for metricFuncLine in metricFuncLines:
					self.logMessage(metricFuncLine)
			else:
				self.logMessage(str(metric))

			if multipleMetrics:
				self.logMessage("Result: ")
				self.logMessage(str(result), False)

		#if extraInfo is not null, we create a new table and add all values in
		#extraInfo
		if extraInfo is not None:
			extraTableHeaders = []
			extraTableValues = []
			for key, value in extraInfo:
				extraTableHeaders.append(str(key))
				extraTableValues.append(str(value))
			extraTable = [extraTableHeaders, extraTableValues]
			self.logMessage(tableString(extraTable, True, None, roundDigits=4))


		#Write the function defining the classifier to the log. If it is
		#a string, write it directly.  Else use inspect to turn the function
		#into a string
		if isinstance(function, (str, unicode)):
			self.logMessage(str(function))
		else:
			funcLines = inspect.getsourcelines(function)
			for funcLine in funcLines:
				self.logMessage(funcLine)


def main():
	trainDataBase = numpy.array([(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)])
	testDataBase = numpy.array([(1.0, 1.0, 1.0), (0.0, 1.0, 0.0)])

	trainData1 = CooSparseData(trainDataBase)
	testData1 = CooSparseData(testDataBase)
	functionStr = """def f():
	return 0"""
	metricsHash = {"rmse":0.50, "meanAbsoluteError":0.45}

	testLogger = HumanReadableRunLog("/Users/rossnoren/UMLMisc/hrTest1.txt")
	testLogger.logTestRun(trainData1, testData1, functionStr, metricsHash)

	functionObj = lambda x: x+1

	testLogger.logTestRun(trainData1, testData1, functionObj, metricsHash)

if __name__ == "__main__":
	main()
