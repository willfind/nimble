# PEP 366 'boilerplate', plus the necessary import of the top level package
if __name__ == "__main__" and __package__ is None:
	import sys
	# add UML parent directory to sys.path
	sys.path.append(sys.path[0].rsplit('/',2)[0])
	import UML
	__package__ = "UML.logging"

import time
import numpy
import inspect

from tableString import *
from logger import Logger
from ..processing.coo_sparse_data import CooSparseData

class HumanReadableRunLog(Logger):

	def __init__(self, logFileName=None):
		super(HumanReadableRunLog, self).__init__(logFileName)


	def logRun(self, trainData, testData, function, metrics, runTime, extraInfo=None):
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

		self.logMessage('*'*80)

		tableHeaders = []
		tableRow = []

		tableHeaders.append("Timestamp")
		tableRow.append(time.strftime('%Y-%m-%d %H:%M:%S'))

		#add number of training points, # of of features to output list
		if trainData.data is not None:
			tableHeaders.append("Train points")
			tableRow.append(str(trainData.data.shape[0]))
			if testData.data is not None:
				tableHeaders.append("Test points")
				tableRow.append(str(testData.data.shape[0]))
				if trainData.data.shape[1] == testData.data.shape[1]:
					tableHeaders.append("Train/Test features")
					tableRow.append(str(trainData.data.shape[1]))
				else:
					tableHeaders.append("Train features")
					tableRow.append(str(trainData.data.shape[1]))
					tableHeaders.append("Test features")
					tableRow.append(str(testData.data.shape[1]))
			else:
				tableHeaders.append("Train features")
				tableRow.append(str(trainData.data.shape[1]))
		else:
			tableHeaders.append("Train points")
			tableHeaders.append("0")

		if runTime is not None:
			tableHeaders.append("Run Time")
			tableRow.append("{0:.2f}".format(runTime))

		#Print table w/basic info to the log
		basicTable = [tableHeaders, tableRow]
		basicTableStr = tableString(basicTable, True, None, roundDigits=4)
		self.logMessage(basicTableStr)

		#if extraInfo is not null, we create a new table and add all values in
		#extraInfo
		if extraInfo is not None:
			extraTableHeaders = []
			extraTableValues = []
			for key, value in extraInfo.iteritems():
				extraTableHeaders.append(str(key))
				extraTableValues.append(str(value))
			extraTable = [extraTableHeaders, extraTableValues]
			self.logMessage(tableString(extraTable, True, None, roundDigits=4))


		#Write the function defining the classifier to the log. If it is
		#a string, write it directly.  Else use inspect to turn the function
		#into a string
		self.logMessage("")
		self.logMessage("Function:")
		if isinstance(function, (str, unicode)):
			self.logMessage(str(function))
		else:
			funcLines = inspect.getsourcelines(function)[0]
			for funcLine in funcLines:
				self.logMessage(str(funcLine))

		#Print out the name/function text of the error metric being used (if there
		#is only one), or the rate & name/function text if more than one is being
		#used
		if metrics is not None:
			metricTable = []
			metricHeaders = []
			metricHeaders.append("\n\nError rate")
			metricHeaders.append("Error Metric")
			metricTable.append(metricHeaders)
			for metric, result in metrics.iteritems():
				metricRow = []
				metricRow.append(str(result))

				if inspect.isfunction(metric):
					metricFuncLines = inspect.getsourcelines(metric)[0]
					metricString = ""
					for metricFuncLine in metricFuncLines:
						metricString += metricFuncLine
				else:
					metricString = str(metric)
				metricRow.append(metricString)
				metricTable.append(metricRow)
			self.logMessage(tableString(metricTable, True, None, roundDigits=4))


def main():
	trainDataBase = numpy.array([(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)])
	testDataBase = numpy.array([(1.0, 1.0, 1.0), (0.0, 1.0, 0.0)])

	trainData1 = CooSparseData(trainDataBase)
	testData1 = CooSparseData(testDataBase)
	functionStr = """def f():
	return 0"""
	metricsHash = {"rmse":0.50, "meanAbsoluteError":0.45}
	extra = {"c":0.5, "folds":10, "tests": 20}

	testLogger = HumanReadableRunLog("/Users/rossnoren/UMLMisc/hrTest1.txt")
	testLogger.logRun(trainData1, testData1, functionStr, metricsHash, extra)

	functionObj = lambda x: x+1

	testLogger.logRun(trainData1, testData1, functionObj, metricsHash, extra)

if __name__ == "__main__":
	main()
