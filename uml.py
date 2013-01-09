"""
Wrapper module, allowing for easy access to the major functions of the package from the
root.

"""

from .interfaces.universal_interface import run as runImplementation
from .processing import CooSparseData
from .processing import DenseMatrixData
from .processing import RowListData
#from .combinations.CrossValidate import crossValidate as cvImplementation


def run(package, algorithm, trainData, testData, output=None, dependentVar=None, arguments={}):
	return runImplementation(package, algorithm, trainData, testData, output, dependentVar, arguments)


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




