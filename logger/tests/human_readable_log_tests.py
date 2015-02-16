
import tempfile
import sys
import os
import numpy

import UML
from UML.logger.human_readable_log import HumanReadableLogger

from UML.tests.testConfig import safetyWrapper



@safetyWrapper
def test_mirrorTostandardOut():
	UML.settings.set('logger', 'mirrorToStandardOut', 'True')

	logTarget = tempfile.NamedTemporaryFile()
	stdoutReplacement = tempfile.NamedTemporaryFile()
	sys.stdout = stdoutReplacement

	try:
		startSize = os.path.getsize(stdoutReplacement.name)

		toTest = HumanReadableLogger(logTarget.name)
		toTest.logMessage("Test Message")

		sys.stdout.flush()

		endSize = os.path.getsize(stdoutReplacement.name)
		assert startSize != endSize
	finally:
		sys.stdout = sys.__stdout__


def TODO_HR_Basic():  # redo test
	trainDataBase = numpy.array([(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0)])
	testDataBase = numpy.array([(1.0, 1.0, 1.0), (0.0, 1.0, 0.0)])

	trainData1 = UML.createData('Sparse', trainDataBase)
	testData1 = UML.createData('Sparse', testDataBase)
	metrics = ["rootMeanSquareError", "meanAbsoluteError"]
	results = [0.50,0.45]
	extra = {"c":0.5, "folds":10, "tests": 20}

	testDir = UML.settings.get("logger", 'location')
	path = os.path.join(testDir, 'hrTest1.txt')
	testLogger = HumanReadableLogger(path)

	functionObj = lambda x: x+1

	testLogger.logRun(trainData1, None, testData1, None, functionObj, metrics, None, results, 0.5, extra)

	# clean up test file
	os.remove(path)
