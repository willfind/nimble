"""
A group of tests which passes data to train, trainAndApply, and trainAndTest, 
checking that after the call, the input data remains unmodified. It makes
use of listLearners and learnerType to try this operation with as many learners
as possible

"""

import UML

from UML.exceptions import ArgumentException

from UML.umlHelpers import generateClusteredPoints
from UML.randomness import pythonRandom

def generateClassificationData():
	"""
	Randomly generate sensible data for a classification problem. Returns a tuple of tuples,
	where the first value is a tuple containing (trainX, trainY) and the second value is
	a tuple containing (testX ,testY)

	"""
	clusterCount = 2
	pointsPer = 10
	featuresPer = 5

	#add noise to the features only
	trainData, trainLabels, noiselessTrainLabels = generateClusteredPoints(clusterCount, pointsPer, featuresPer, addFeatureNoise=True, addLabelNoise=False, addLabelColumn=False)
	testData, testLabels, noiselessTestLabels = generateClusteredPoints(clusterCount, 1, featuresPer, addFeatureNoise=True, addLabelNoise=False, addLabelColumn=False)

	return ((trainData, noiselessTrainLabels), (testData, noiselessTestLabels))

def generateRegressionData():
	"""
	Randomly generate sensible data for a regression problem. Returns a tuple of tuples,
	where the first value is a tuple containing (trainX, trainY) and the second value is
	a tuple containing (testX ,testY)

	"""
	clusterCount = 2
	pointsPer = 10
	featuresPer = 5

	#add noise to both the features and the labels
	regressorTrainData, trainLabels, noiselessTrainLabels = generateClusteredPoints(clusterCount, pointsPer, featuresPer, addFeatureNoise=True, addLabelNoise=True, addLabelColumn=False)
	regressorTestData, testLabels, noiselessTestLabels = generateClusteredPoints(clusterCount, 1, featuresPer, addFeatureNoise=True, addLabelNoise=True, addLabelColumn=False)

	return ((regressorTrainData, trainLabels), (regressorTestData, testLabels))


def assertUnchanged(learnerName, passed, trainX, trainY, testX, testY):
	"""
	Helper to assert that those objects passed down into a function are
	identical to copies made before the call 
	"""
	((pTrainX, pTrainY),(pTestX, pTestY)) = passed

	if not pTrainX.isIdentical(trainX):
		raise ValueError(learnerName + " modified its trainX data")
	if not pTrainY.isIdentical(trainY):
		raise ValueError(learnerName + " modified its trainY data")
	if not pTestX.isIdentical(testX):
		raise ValueError(learnerName + " modified its testX data")
	if not pTestY.isIdentical(testY):
		raise ValueError(learnerName + " modified its testY data")

def wrappedTrain(learnerName, trainX, trainY, testX, testY):
	return UML.train(learnerName, trainX, trainY)

def wrappedTrainAndApply(learnerName, trainX, trainY, testX, testY):
	return UML.trainAndApply(learnerName, trainX, trainY, testX)

def wrappedTrainAndApplyOvO(learnerName, trainX, trainY, testX, testY):
	return UML.umlHelpers.trainAndApplyOneVsOne(learnerName, trainX, trainY, testX)

def wrappedTrainAndApplyOvA(learnerName, trainX, trainY, testX, testY):
	return UML.umlHelpers.trainAndApplyOneVsAll(learnerName, trainX, trainY, testX)

def wrappedTrainAndTest(learnerName, trainX, trainY, testX, testY):
	# our performance function doesn't actually matter, we're just checking the data
	return UML.trainAndTest(learnerName, trainX, trainY, testX, testY, performanceFunction=UML.metrics.fractionIncorrect)

def wrappedTrainAndTestOvO(learnerName, trainX, trainY, testX, testY):
	return UML.umlHelpers.trainAndTestOneVsOne(learnerName, trainX, trainY, testX, testY, performanceFunction=UML.metrics.fractionIncorrect)

def wrappedTrainAndTestOvA(learnerName, trainX, trainY, testX, testY):
	return UML.umlHelpers.trainAndTestOneVsAll(learnerName, trainX, trainY, testX, testY, performanceFunction=UML.metrics.fractionIncorrect)


def backend(toCall, portionToTest):
	cData = generateClassificationData()
	((cTrainX, cTrainY), (cTestX, cTestY)) = cData
	backCTrainX = cTrainX.copy()
	backCTrainY = cTrainY.copy()
	backCTestX = cTestX.copy()
	backCTestY = cTestY.copy()
	rData = generateRegressionData()
	((rTrainX, rTrainY), (rTestX, rTestY)) = rData
	backRTrainX = rTrainX.copy()
	backRTrainY = rTrainY.copy()
	backRTestX = rTestX.copy()
	backRTestY = rTestY.copy()

	allLearners = UML.listLearners()
	numSamples = int(len(allLearners) * portionToTest)
	toTest = pythonRandom.sample(allLearners, numSamples)

	for learner in toTest:
		package = learner.split('.',1)[0].lower()
		if package != 'mlpy' and package != 'scikitlearn':
			continue 
		lType = UML.learnerType(learner)
		if lType == 'classification':
			try:
				toCall(learner, cTrainX, cTrainY, cTestX, cTestY)
			# this is meant to safely bypass those learners that have required arguments
			except ArgumentException as ae:
				print ae
			assertUnchanged(learner, cData, backCTrainX, backCTrainY, backCTestX, backCTestY)
		if lType == 'regression':
			try:
				toCall(learner, rTrainX, rTrainY, rTestX, rTestY)
			# this is meant to safely bypass those learners that have required arguments
			except ArgumentException as ae:
				print ae
			assertUnchanged(learner, rData, backRTrainX, backRTrainY, backRTestX, backRTestY)

def testDataIntegretyTrain():
	backend(wrappedTrain, 1)

def testDataIntegretyTrainAndApply():
	backend(wrappedTrainAndApply, 1)

# we can test smaller portions here because the backends are all being tested by
# the previous tests. We only care about the trainAndApply One vs One and One vs
# all code.
def testDataIntegretyTrainAndApplyMulticlassStrategies():
	backend(wrappedTrainAndApplyOvO, .05)
	backend(wrappedTrainAndApplyOvA, .05)

def testDataIntegretyTrainAndTest():
	backend(wrappedTrainAndTest, 1)

# we can test smaller portions here because the backends are all being tested by
# the previous tests. We only care about the trainAndTest One vs One and One vs
# all code.
def testDataIntegretyTrainAndTestMulticlassStrategies():
	backend(wrappedTrainAndTestOvO, .05)
	backend(wrappedTrainAndTestOvA, .05)






