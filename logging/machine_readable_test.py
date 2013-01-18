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
import inspect
import re
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

		#Create a string to be logged (as one line), and add dimensions and the function
		logLine = ""
		logLine += createMRLineElement("time", str(datetime.datetime.now()))
		logLine += createMRLineElement("numTrainDataPoints", str(trainData.data.shape[0]))
		logLine += createMRLineElement("numTrainDataFeatures", str(trainData.data.shape[1]))
		logLine += createMRLineElement("numTestDataPoints", str(testData.data.shape[0]))
		logLine += createMRLineElement("numTestDataFeatures", str(testData.data.shape[1]))

		if isinstance(function, (str, unicode)):
			logLine += createMRLineElement("function", function)
		else:
			#we get the source code of the function as a list of strings and glue them together
			funcLines = inspect.getsourcelines(function)
			funcString = ""
			for i in range(len(funcLines) - 1):
				funcString += str(funcLines[i])
			if funcLines is None:
				funcLines = "N/A"
			logLine += createMRLineElement("function", funcString)

		#add any extraInfo to the log string
		if extraInfo is not None:
			for key, value in extraInfo.items():
				logLine += createMRLineElement(key, str(value))

		for metric, result in metrics.items():
			logLine += createMRLineElement(str(metric), str("{0:.4f}").format(result))

		if logLine[len(logLine)-1] == ',':
			logLine = logLine[:-1]

		self.logMessage(logLine)

def parseLog(pathToLogFile):
	"""
		TODO: add docstring
	"""
	logFile = open(pathToLogFile, 'r')
	rawRuns = logFile.readLines()
	parsedRuns = []
	for rawRun in rawRuns:
		run = parseLoggedRun(rawRun)
		parsedRuns.append(run)

	return parsedRuns

def parseLoggedRun(loggedRun):
	"""
		TODO: add docstring
	"""
	runDict = {}
	elements = re.split(r"[^\\],", loggedRun)
	for element in elements:
		parts = re.split(r"([^\\]\\\\|[^\\]):", element)
		#we expect that each element has two parts (they should be of the form
		#key:value), so if there are more or fewer than 2, we raise an exception
		if len(parts):
			raise Exception("Badly formed line in log of runs")
		runDict[parts[0]] = parts[1]

	return runDict

def createMRLineElement(key, value, addComma=True):
	"""
		TODO: add docstring
	"""
	processedValue = sanitizeStringForLog(value)
	result = key+":\""+processedValue+"\""
	if addComma:
		result += ","

	return result

#TODO fill out body of function
def unSanitizeStringFromLog(sanitizedString):
	"""
		TODO: add docstring
	"""
	return

def sanitizeStringForLog(rawString):
	"""
		Escape all characters in rawString that may interfere with the machine-readable
		logging format: double-quotes, commas, colons, carriage returns, line-feeds, and
		backslashes.  Takes a string as argument; returns that string with an additional
		backslash in front of any of the above-mentioned special characters.
	"""

	if rawString is None or rawString == "":
		return

	rawString = rawString.replace("\\", "\\\\")
	rawString = rawString.replace("\"", "\\\"")
	rawString = rawString.replace("\n", "\\n")
	rawString = rawString.replace("\r", "\\r")
	rawString = rawString.replace(":", "\:")
	rawString = rawString.replace(",", "\,")

	return rawString


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
