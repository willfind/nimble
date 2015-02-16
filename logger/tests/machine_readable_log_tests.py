
import inspect
import re
import numpy
import os

import UML
from UML.logger.machine_readable_log import MachineReadableLogger
from UML.logger.machine_readable_log import sanitizeStringForLog
from UML.logger.machine_readable_log import unSanitizeStringFromLog
from UML.logger.machine_readable_log import createMRLineElement
from UML.logger.machine_readable_log import parseLog
from UML.logger.stopwatch import Stopwatch

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

	trainData1 = UML.createData('Sparse', trainDataBase)
	testData1 = UML.createData('Sparse', testDataBase)
	functionStr = """def f():
	return 0"""
	metrics = ["rootMeanSquareError", "meanAbsoluteError"]
	results = [0.50, 0.45]

	testDir = UML.settings.get("logger", 'location')
	logpath = os.path.join(testDir, 'mrTest2.txt')
	testLogger = MachineReadableLogger(logpath)
	testLogger.logRun(trainData1, None, testData1, None, functionStr, metrics, None, results, Stopwatch())

	functionObj = lambda x: x+1

	testLogger.logRun(trainData1, None, testData1, None, functionObj, metrics, None, results, Stopwatch())

	logDicts = parseLog(logpath)

	assert logDicts[0]["numTrainDataPoints"] == '3'
	assert logDicts[0]["numTestDataPoints"] == '2'
	assert logDicts[0]["numTrainDataFeatures"] == '3'
	assert logDicts[0]["numTestDataFeatures"] == '3'
#	assert logDicts[0]["runTime"] == '0.00'
	assert logDicts[0]["function"] == 'def f():\n\treturn 0'
	assert logDicts[0]["rootMeanSquareError"] == '0.5'
	assert logDicts[0]["meanAbsoluteError"] == '0.45'

	assert logDicts[1]["numTrainDataPoints"] == '3'
	assert logDicts[1]["numTestDataPoints"] == '2'
	assert logDicts[1]["numTrainDataFeatures"] == '3'
	assert logDicts[1]["numTestDataFeatures"] == '3'
#	assert logDicts[1]["runTime"] == '0.00'
	assert logDicts[1]["function"] == "['\\tfunctionObj = lambda x: x+1\n']"
	assert logDicts[1]["rootMeanSquareError"] == '0.5'
	assert logDicts[1]["meanAbsoluteError"] == '0.45'

	# clean up test file
	os.remove(logpath)


def TODO_MR_basic():  # redo unit test
	trainDataBase = numpy.array([(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)])
	testDataBase = numpy.array([(1.0, 1.0, 1.0), (0.0, 1.0, 0.0)])

	trainData1 = UML.createData('Sparse', trainDataBase)
	testData1 = UML.createData('Sparse', testDataBase)
	functionStr = """def f():
	return 0"""
	metrics = ["rootMeanSquareError", "meanAbsoluteError"]
	results = [0.50, 0.45]

	testDir = UML.settings.get("logger", 'location')
	logpath = os.path.join(testDir, 'mrTest1.txt')
	testLogger = MachineReadableLogger(logpath)
	testLogger.logRun(trainData1, None, testData1, None, functionStr, metrics, None, results, 0.0)

	functionObj = lambda x: x+1

	testLogger.logRun(trainData1, None, testData1, None, functionObj, metrics, None, results, 0.0)

	# clean up test file
	os.remove(logpath)

	assert False
