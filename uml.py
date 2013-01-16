"""
Wrapper module, allowing for easy access to the major functions of the package from the
root.

"""

from .interfaces import mahout
from .interfaces import regressor
from .interfaces import sciKitLearn
from .interfaces import mlpy
from .processing import CooSparseData
from .processing import DenseMatrixData
from .processing import RowListData
#from .combinations.CrossValidate import crossValidate as cvImplementation


def run(package, algorithm, trainData, testData, output=None, dependentVar=None, arguments={}):
	if package == 'mahout':
		return mahout(algorithm, trainData, testData, output, dependentVar, arguments)
	if package == 'regressor':
		return regressors(algorithm, trainData, testData, output, dependentVar, arguments)
	if package == 'sciKitLearn':
		return sciKitLearn(algorithm, trainData, testData, output, dependentVar, arguments)
	if package == 'mlpy':
		return mlpy(algorithm, trainData, testData, output, dependentVar, arguments)


# run() with a return type of the predicted labels added back into the object?


def normalize(package, algorithm, trainData, testData, dependentVar=None, arguments={}, mode=True):
	"""
	Calls on the functionality of a package to train on some data and then modify both
	the training data and a set of test data accroding to the produced model.


	"""
	# single call normalize, combined data
	if mode:
		testLength = testData.points()
		# glue training data at the end of test data
		testData.appendPoints(trainData)
		normalizedAll = run(package, algorithm, trainData, testData, dependentVar=dependentVar, arguments=arguments)
		# resplit normalized
		normalizedTrain = normalizedAll.extractPoints(start=testLength, end=normalizedAll.points())
		normalizedTest = normalizedAll
	# two call normalize, no data combination
	else:
		normalizedTrain = run(package, algorithm, trainData, trainData, dependentVar=dependentVar, arguments=arguments)
		normalizedTest = run(package, algorithm, trainData, testData, dependentVar=dependentVar, arguments=arguments)
		
	# modify references for trainData and testData
	trainData.copyReferences(normalizedTrain)
	testData.copyReferences(normalizedTest)


def data(dataType, data=None, featureNames=None):
	if dataType == "CooSparseData":
		return CooSparseData(data, featureNames)
	elif dataType == "DenseMatrixData":
		return DenseMatrixData(data, featureNames)
	elif dataType == "RowListData":
		return RowListData(data, featureNames)
	else:
		raise ArgumenException("Unknown data type, cannot instantiate")


# def runWithPerformance()  # same as run, with extra parameter?


def crossValidate(X, Y, functionsToApply, numFolds=10):
	return cvImplementation(X, Y, functionsToApply, numFolds)


#combinations() -- maybe

#listAllAlgorithms()




