# PEP 366 'boilerplate', plus the necessary import of the top level package
if __name__ == "__main__" and __package__ is None:
	import sys
	# add UML parent directory to sys.path
	sys.path.append(sys.path[0].rsplit('/',2)[0])
	import UML
	import UML.logging
	__package__ = "UML.logger"

import datetime
import numpy
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
		else: pass

		self.logMessage('*'*80)
		self.logMessage(str(datetime.datetime.now()))
		
		self.logMessage('Number of training points: ')
		self.logMessage(str(trainData.data.shape[0]), addNewLine=False)
		self.logMessage('Number of testing points: ')
		self.logMessage(str(testData.data.shape[0]), addNewLine=False)
		self.logMessage('Number of training features: ')
		self.logMessage(str(trainData.data.shape[1]), addNewLine=False)
		self.logMessage('Number of testing features: ')
		self.logMessage(str(testData.data.shape[1]), addNewLine=False)
		
		#Write the function defining the classifier to the log. If it is
		#a string, write it directly.  Else use pickle to turn the function
		#into a string
		self.logMessage('Classifier Training Function: ')
		if isinstance(function, (str, unicode)):
			self.logMessage(function)
		else:
			self.logMessage(repr(function))
		self.logMessage('\n', False)

		#if extraInfo is not null, we print whatever is in it in the form key: value
		if extraInfo is not None:
			for key, value in extraInfo.items():
				message = str(key) + ": " + str(value)
				self.logMessage(message)
			self.logMessage('\n', False)

		#Write performance metrics
		for metric, result in metrics.items():
			self.logMessage("METRIC FUNCTION: ")
			if isinstance(metric, (str, unicode)):
				self.logMessage(metric)
			else:
				self.logMessage(repr(metric))

			self.logMessage("RESULT: {0:.4f}".format(result))

		self.logMessage('\n', False)
		self.logMessage('*'*80)


#	def write(self, message, addNewLine=True):
#		"""
#			Helper function to make things cleaner: instead of calling self.logMessage(),
#			we can just call write()
#		"""
#		self.logMessage(message, addNewLine)

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
