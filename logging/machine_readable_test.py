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

class MachineReadableRunLog(Logger):

	def __init__(self, logFileName=None):
		Logger.__init__(self, logFileName)

	def logTestRun(self, trainData, testData, function, metrics, extraInfo=None):
		"""
			Write one (data + classifer + error metrics) combination to a log file
			in machine readable format.  Should include as much information as possible,
			to allow someone to reproduce the test.  Information included:
			# of training data points
			# of testing data points
			# of features in training data
			# of features in testing data
			Function defining the classifer (algorithm, parameters, etc.)
			Error metrics computed based on predictions of classifier: name/function and numerical
			result)
			
			Format is key:value,key:value,...,key:value
		"""
		self.logMessage("Time:"+str(datetime.datetime.now())+",", False)
		self.logMessage("numTrainDataPoints:"+str(trainData.data.shape[0])+",", False)
		self.logMessage("numTrainDataFeatures:"+str(trainData.data.shape[1])+",", False)
		self.logMessage("numTestDataPoints:"+str(testData.data.shape[0])+",", False)
		self.logMessage("numTestDataFeatures:"+str(testData.data.shape[1])+",", False)
		self.logMessage("function:"+str(function)+",", False)

		if extraInfo is not None:
			for key, value in extraInfo.items():
				self.logMessage(str(key)+":"+str(value)+",", False)

		for metric, result in metrics.items():
			self.logMessage(metric+":"+str("{0:.4f}".format(result))+",", False)


def main():
	trainDataBase = numpy.array([(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)])
	testDataBase = numpy.array([(1.0, 1.0, 1.0), (0.0, 1.0, 0.0)])

	trainData1 = CooSparseData(trainDataBase)
	testData1 = CooSparseData(testDataBase)
	functionStr = """def f():
	return 0"""
	metricsHash = {"rmse":0.50, "meanAbsoluteError":0.45}

	testLogger = MachineReadableRunLog("/Users/rossnoren/UMLMisc/mrTest1.txt")
	testLogger.logTestRun(trainData1, testData1, functionStr, metricsHash)

	functionObj = lambda x: x+1

	testLogger.logTestRun(trainData1, testData1, functionObj, metricsHash)

if __name__ == "__main__":
	main()