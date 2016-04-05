
import tempfile
import sys
import os
import numpy
import time
import StringIO

import UML
from UML.customLearners import CustomLearner
from UML.logger.human_readable_log import HumanReadableLogger
from UML.configuration import configSafetyWrapper
from UML.calculate import fractionIncorrect


@configSafetyWrapper
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

@configSafetyWrapper
def test_HR_logger_output_fromFile_trainAndApply():
	UML.settings.set('logger', 'mirrorToStandardOut', 'True')
	
	trainData = [[1,2,3], [4,5,6], [7,8,9]]
	trainOrig = UML.createData(returnType="Matrix", data=trainData)

	trainLab = UML.createData(returnType="Matrix", data=[[1],[0],[1]])
	testOrig = UML.createData(returnType="Matrix", data=[[10,11,12]])

	buf = StringIO.StringIO()
	savedOut = sys.stdout
	sys.stdout = buf

	# instantiate from csv file
	with tempfile.NamedTemporaryFile(suffix=".csv") as tmpCSV:
		trainOrig.writeFile(tmpCSV.name)

		fromFileTrain = UML.createData(returnType="Matrix", data=tmpCSV.name)

		UML.trainAndApply("custom.RidgeRegression",fromFileTrain, trainLab, testOrig)

	sys.stdout = savedOut

	fullOutput = buf.getvalue()

#	print fullOutput.split('\n')

	assert False  # incomplete. revise when logger output is finalized

	#time.strptime("2015-03-30 13:38:15", '%Y-%m-%d %H:%M:%S')


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


class UnitPred(CustomLearner):
	learnerType = "classification"

	def train(self, trainX, trainY):
		return

	def apply(self, testX):
		return testX.copy()

def testFractionIncorrectCITriggering():
	back_CI_triggering("fractionIncorrect")

def testRMSECITriggering():
	back_CI_triggering("rootMeanSquareError")

def testMeanAbsoluteErrorCITriggering():
	back_CI_triggering("meanAbsoluteError")

@configSafetyWrapper
#def back_CI_triggering(perfFuncName, triggerType):
def back_CI_triggering(perfFuncName):
	perfFunc = getattr(UML.calculate, perfFuncName)
	CIName = perfFuncName + "ConfidenceInterval"
	CIGen = getattr(UML.calculate.confidence, CIName)

	trainX = UML.createData("List", [1,1,1,1,1,0,0,0,0,0])
	trainX.transpose()
	trainY = trainX.copy()
	testX = UML.createData("List", [1,1,1,1,0,1,0,0,0,0])
	testX.transpose()
	testY = trainY.copy()

	UML.registerCustomLearner('custom', UnitPred)

	def logTriggerTrainAndTest(trainX, trainY, testX, testY):
		UML.trainAndTest(
			"custom.UnitPred", trainX, trainY,testX, testY,
			performanceFunction=perfFunc)

	def logTriggerTLTest(trainX, trainY, testX, testY):
		tl = UML.train("custom.UnitPred", trainX, trainY)
		tl.test(testX, testY, performanceFunction=perfFunc)

	def logTriggerTrainAndTestOnTrainingData(trainX, trainY, testX, testY):
		UML.trainAndTestOnTrainingData(
			"custom.UnitPred", trainX, trainY,performanceFunction=perfFunc)

	def logTriggerTrainAndTestOneVsOne(trainX, trainY, testX, testY):
		UML.helpers.trainAndTestOneVsOne(
			"custom.UnitPred", trainX, trainY,testX, testY,
			performanceFunction=perfFunc)

	def logTriggerTrainAndTestOneVsAll(trainX, trainY, testX, testY):
		UML.helpers.trainAndTestOneVsAll(
			"custom.UnitPred", trainX, trainY,testX, testY,
			performanceFunction=perfFunc)

	possible = [logTriggerTrainAndTest, logTriggerTLTest,
		logTriggerTrainAndTestOnTrainingData, logTriggerTrainAndTestOneVsOne,
		logTriggerTrainAndTestOneVsAll]

	for triggerFunc in possible:
		(start, end) = runAndLogCheck(triggerFunc, trainX, trainY, testX, testY)
		lengthWithCI = end - start

		# cripple CI, run the second trial
		try:
			delattr(UML.calculate.confidence, CIName)
			(start, end) = runAndLogCheck(triggerFunc, trainX, trainY, testX, testY)
			lengthWithoutCI = end - start
		finally:
			setattr(UML.calculate.confidence, CIName, CIGen)

		# strictly less than to indicate that something was printed for CI
		assert lengthWithoutCI < lengthWithCI

	UML.deregisterCustomLearner("custom", "UnitPred")



def runAndLogCheck(toCall, trainX, trainY, testX, testY):
	"""
	Call the given function with the given arguments while keeping
	track of the size of the log file before and after the call

	"""
	# log file path
	loc = UML.settings.get('logger', 'location')
	name = UML.settings.get('logger', 'name')
	# could check human readable or machine readable. we choose HR only,
	# thus the addition of .txt
	path = os.path.join(loc, name + '.txt')

	if os.path.exists(path):
		startSize = os.path.getsize(path)
	else:

		startSize = 0

	# run given function
	toCall(trainX, trainY, testX, testY)

	# make sure it has the expected effect on the size
	if os.path.exists(path):
		endSize = os.path.getsize(path)
	else:
		endSize = 0

	return (startSize, endSize)
