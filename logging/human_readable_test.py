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


	def logTestRun(self, trainData, testData, function, metrics, extraInfo=None):
		"""
			Write one (data + classifer + error metrics) combination to a log file
			in a human readable format.  Should include as much information as possible,
			to allow someone to reproduce the test.  Information included:
			# of training data points
			# of testing data points
			# of features in training data
			# of features in testing data
			Function defining the classifer (algorithm, parameters, etc.)
			Error metrics computed based on predictions of classifier: name/function and numerical
			result)
		"""
		#if the log file is not available, try to create it
		if not self.isAvailable:
			self.setup()

		tableRow = []

		tableRow.append(str(datetime.datetime.now()))

		#add # of training points to output list
		if trainData.data is not None:
			tableRow.append(str(trainData.data.shape[0]))
			tableRow.append(str(trainData.data.shape[1]))
		else:
			tableRow.append("")
			tableRow.append("")
		
		#add # of testing points to output list
		if testData.data is not None:
			tableRow.append(str(testData.data.shape[0]))
			tableRow.append(str(testData.data.shape[1]))
		else:
			tableRow.append("")
			tableRow.append("")

		
		#Write the function defining the classifier to the log. If it is
		#a string, write it directly.  Else use inspect to turn the function
		#into a string
		if isinstance(function, (str, unicode)):
			tableRow.append(function)
		else:
			tableRow.append(inspect.getsourcelines(function))

		#Write performance metrics
		for metric in metrics:
			tableRow.append(str(metrics[metric]))

		#if extraInfo is not null, we append all values in the dictionary to tableRow list
		if extraInfo is not None:
			for key, value in extraInfo:
				tableRow.append(str(value))


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
