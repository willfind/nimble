
from allowImports import boilerplate
boilerplate()
import random
from UML import *
from UML.metrics import rmse, meanAbsoluteError


if __name__ == "__main__":

	#fileName = "../datasets/noisy-linear.csv"
	fileName = "../datasets/concrete_slump.csv"
	trainX, trainY, testX, testY = loadTrainingAndTesting(fileName, labelID='Compressive Strength', fractionForTestSet=.15, fileType="csv")
	#random.seed = 5
	#remove this column, since it's just the ID number of each data point
	trainX.extractFeatures("No")
	testX.extractFeatures("No")



	#do some dimensionality reduction
	normalize('mlpy.PCA', trainX, testX, arguments={'k':5})

	"""
	results = runAndTestDirect("mlpy.Ridge", trainX, testX, trainY, testY, arguments={"lmb":1}, performanceMetricFuncs=[rmse, meanAbsoluteError])

	print "results", results
	"""


	toRun = 'runAndTestDirect("mlpy.Ridge", trainX, testX, trainY, testY, {"lmb":<.01|.1|1>}, [rmse])'
	runs = functionCombinations(toRun)
	runs.append('runAndTestDirect("mlpy.LARS", trainX, testX, trainY, testY, {"maxsteps":20}, [rmse])')

	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=10, extraParams=locals())


	print "bestFunction:\n", bestFunction
	print "performance: ", performance






	#OLD STUFF

	##results = runAndTestDirect("mlpy.LibSvm", trainX, testX, trainY, testY, arguments={"C":1}, performanceMetricFuncs=[classificationError])

	#xData = data("DenseMatrixData", fileName, fileType="csv")
	#trainX, trainY, testX, testY = loadTrainingAndTesting(fileName, labelID='y', fractionForTestSet=.15, fileType="csv")
	#normalize('mlpy.PCA', trainX, testX, arguments={'k':2})



	#split out the label for this data
	#yData = matrix.extractFeatures("Compressive Strength")

	#get rid of feature "No" column since it's just gives each data point a number
	#xData.extractFeatures("No")

	#split into training and testing