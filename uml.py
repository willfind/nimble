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




