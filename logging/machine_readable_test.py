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
import re
from logger import Logger
from ..processing.coo_sparse_data import CooSparseData

class MachineReadableRunLog(Logger):

	def __init__(self, logFileName=None):
		Logger.__init__(self, logFileName)

	def logRun(self, trainData, testData, function, metrics, runTime, extraInfo=None):
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
			Any additional information, definedy by user, passed as 'extraInfo'
			
			Format is key:value,key:value,...,key:value
		"""

		#Create a string to be logged (as one line), and add dimensions and the function
		logLine = "{RUN}::"
		logLine += createMRLineElement("timestamp", time.strftime('%Y-%m-%d %H:%M:%S'))
		logLine += createMRLineElement("numTrainDataPoints", trainData.data.shape[0])
		logLine += createMRLineElement("numTrainDataFeatures", trainData.data.shape[1])
		logLine += createMRLineElement("numTestDataPoints", testData.data.shape[0])
		logLine += createMRLineElement("numTestDataFeatures", testData.data.shape[1])
		logLine += createMRLineElement("runTime", "{0:.2f}".format(runTime))

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
				logLine += createMRLineElement(key, value)

		if metrics is not None:
			for metric, result in metrics.items():
				if isinstance(metric, (str, unicode)):
					logLine += createMRLineElement(str(metric), result)
				else:
					metricLines = inspect.getsourcelines(metric)
					metricString = ""
					for i in range(len(metricLines) - 1):
						metricString += str(metricLines[i])
					if metricLines is None:
						metricLines = "N/A"
					logLine += createMRLineElement(metricString, result)

		if logLine[len(logLine)-1] == ',':
			logLine = logLine[:-1]

		self.logMessage(logLine)

def parseLog(pathToLogFile):
	"""
		Provided with a path to a log file containing some lines representing machine-readable
		logs of runs, read all such lines and parse them into a list of hashes.  Each hash
		represents one run, and contains the logged information about that run.  Assumes that
		each line in the log file that starts with {RUN} represents a run; any line that doesn't
		start with {RUN} is ignored.
	"""
	logFile = open(pathToLogFile, 'r')
	rawRuns = logFile.readlines()
	parsedRuns = []
	for rawRun in rawRuns:
		if rawRun.startswith("{RUN}::"):
			rawRun = rawRun.replace("{RUN}::", "", 1)
			run = parseLoggedRun(rawRun)
			parsedRuns.append(run)

	return parsedRuns

def parseLoggedRun(loggedRun):
	"""
		Convert one line of a log file - which represents output information of one run - into
		a dictionary containing the same information, keyed by standard labels (see
		MachineReadableRunLog.logRun() for examples of labels that are used).
	"""
	runDict = {}
	#split each logged Run wherever there is a comma not preceded by a backslash
	elements = re.split(r"[^\\],", loggedRun)
	for element in elements:
		parts = re.split(r"[^\\](\\\\)*:", element)
		#we expect that each element has two parts (they should be of the form
		#key:value), so if there are more or fewer than 2, we raise an exception
		if len(parts) != 2:
			raise Exception("Badly formed line in log of runs")

		key = parts[0]
		value = parts[1]
		unSanitizedValue = unSanitizeStringFromLog(value)
		runDict[key] = unSanitizedValue

	return runDict

def createMRLineElement(key, value, addComma=True):
	"""
		Combine the key and value into a string suitable for a machine-readable log.
		Returns string of the format 'key:"value",' though the double-quotes around
		value are not included if value is a number.  The trailing comma is added by
		default but will be ommitted if addComma is False.
	"""
	if isinstance(value, (bool, int, long, float)):
		processedValue = str(value)
	else:
		processedValue = "\""+sanitizeStringForLog(value)+"\""
	result = key+":"+processedValue
	if addComma:
		result += ","

	return result

def unSanitizeStringFromLog(sanitizedString):
	"""
		Replace escaped versions of characters within sanitizedString with the original,
		unescaped version.  Mirror opposite of sanitizeStringForLog: where sanitize
		replaces newLines with '\n', unSanitize replaces '\n' with a newline.
	"""
	if len(sanitizedString) < 2:
		return sanitizedString

	unsanitizedString = sanitizedString

	if re.search(r"[^\\](\\\\)*\\\\$", unsanitizedString) != None:
		unsanitizedString = unsanitizedString[:-1]

	done = False
	while not done:
		if re.search(r"([^\\]|^)((\\\\)*)(\\n)", unsanitizedString) is not None:
			unsanitizedString = re.sub(r"([^\\]|^)((\\\\)*)(\\)(n)", '\g<1>\g<2>\n', unsanitizedString, 1)
		else:
			done = True

	done = False
	while not done:
		if re.search(r"([^\\]|^)((\\\\)*)(\\r)", unsanitizedString) is not None:
			unsanitizedString = re.sub(r"([^\\]|^)((\\\\)*)(\\)(r)", '\g<1>\g<2>\r', unsanitizedString, 1)
		else:
			done = True

	done = False
	while not done:
		if re.search(r"([^\\]|^)((\\\\)*)(\\)[\":,]", unsanitizedString) is not None:
			unsanitizedString = re.sub(r"([^\\]|^)((\\\\)*)(\\)([\":,])", '\g<1>\g<2>\g<5>', unsanitizedString, 1)
		else:
			done = True

	return unsanitizedString

def sanitizeStringForLog(rawString):
	"""
		Escape all characters in rawString that may interfere with the machine-readable
		logging format: double-quotes, commas, colons, carriage returns, line-feeds, and
		backslashes.  Takes a string as argument; returns that string with an additional
		backslash in front of any of the above-mentioned special characters.
	"""

	if rawString is None or rawString == "":
		return ""

	sanitizedString = rawString
	#escape trailing backslash, if it is not already escaped
	if re.search(r"[^\\](\\\\)*\\$", sanitizedString) != None:
		sanitizedString = sanitizedString + '\\'

	#add preceding backslash to all special characters
	sanitizedString = sanitizedString.replace("\"", "\\\"")
	sanitizedString = sanitizedString.replace("\n", "\\n")
	sanitizedString = sanitizedString.replace("\r", "\\r")
	sanitizedString = sanitizedString.replace(":", "\:")
	sanitizedString = sanitizedString.replace(",", "\,")

	return sanitizedString

def testSanitization():
	"""
		Test the process of sanitizing and unsanitizing strings before/after
		writing to log.  Strings are not required to be exactly the same before
		and after, but should be functionally the same
	"""
	testString1 = """Hello, we've got: fifteen dogs.
	How many dogs have you got?\r"""

	sanitizedString1 = sanitizeStringForLog(testString1)
	unsanitizedString1 = unSanitizeStringFromLog(sanitizedString1)
	assert testString1 == unsanitizedString1

	subFuncLines = inspect.getsourcelines(re.sub)[0]
	for line in subFuncLines:
		assert line == unSanitizeStringFromLog(sanitizeStringForLog(line))

	testString2 = """Hello \n\r where are you: spain?\n we have been to france, germany, and holland.\n\\"""
	assert testString2 == unSanitizeStringFromLog(sanitizeStringForLog(testString2))


def testCreateMRElement():
	"""
		Test(s) for createMRElement() function
	"""
	key1 = 'dog'
	value1 = 'cat'
	key2 = 'pants'
	value2 = 5

	assert 'dog:"cat"' == createMRLineElement(key1, value1, False)
	assert 'dog:"cat",' == createMRLineElement(key1, value1, True)
	assert 'pants:5' == createMRLineElement(key2, value2, False)
	assert 'pants:5,' == createMRLineElement(key2, value2, True)
	assert 'pants:""' == createMRLineElement(key2, '', False)
	assert 'pants:"",' == createMRLineElement(key2, '', True)

def testParseLog():
	"""
		Test the function that reads in machine-readable log files and turns
		them into lists of dictionaries.
	"""
	trainDataBase = numpy.array([(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)])
	testDataBase = numpy.array([(1.0, 1.0, 1.0), (0.0, 1.0, 0.0)])

	trainData1 = CooSparseData(trainDataBase)
	testData1 = CooSparseData(testDataBase)
	functionStr = """def f():
	return 0"""
	metricsHash = {"rmse":0.50, "meanAbsoluteError":0.45}

	testLogger = MachineReadableRunLog("/Users/rossnoren/UMLMisc/mrTest2.txt")
	testLogger.logRun(trainData1, testData1, functionStr, metricsHash, 0.0)

	functionObj = lambda x: x+1

	testLogger.logRun(trainData1, testData1, functionObj, metricsHash, 0.0)

	logDicts = parseLog("/Users/rossnoren/UMLMisc/mrTest2.txt")

	assert logDicts[0]["numTrainDataPoints"] == '3'
	assert logDicts[0]["numTestDataPoints"] == '2'
	assert logDicts[0]["numTrainDataFeatures"] == '3'
	assert logDicts[0]["numTestDataFeatures"] == '3'
	assert logDicts[0]["runTime"] == '0.0'
	assert logDicts[0]["function"] == 'def f():\n\treturn 0'
	assert logDicts[0]["rmse"] == '0.5'
	assert logDicts[0]["meanAbsoluteError"] == '0.45'

	assert logDicts[1]["numTrainDataPoints"] == '3'
	assert logDicts[1]["numTestDataPoints"] == '2'
	assert logDicts[1]["numTrainDataFeatures"] == '3'
	assert logDicts[1]["numTestDataFeatures"] == '3'
	assert logDicts[1]["runTime"] == '0.0'
	assert logDicts[1]["function"] == "['\tfunctionObj = lambda x: x+1\n']"
	assert logDicts[1]["rmse"] == '0.5'
	assert logDicts[1]["meanAbsoluteError"] == '0.45'



def main():
	trainDataBase = numpy.array([(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)])
	testDataBase = numpy.array([(1.0, 1.0, 1.0), (0.0, 1.0, 0.0)])

	trainData1 = CooSparseData(trainDataBase)
	testData1 = CooSparseData(testDataBase)
	functionStr = """def f():
	return 0"""
	metricsHash = {"rmse":0.50, "meanAbsoluteError":0.45}

	testLogger = MachineReadableRunLog("~/mrTest2.txt")
	testLogger.logRun(trainData1, testData1, functionStr, metricsHash, 0.0)

	functionObj = lambda x: x+1

	testLogger.logRun(trainData1, testData1, functionObj, metricsHash, 0.0)



if __name__ == "__main__":
	main()
