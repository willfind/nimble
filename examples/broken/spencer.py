
from allowImports import boilerplate
boilerplate()

import os.path
import UML
from UML import *
from UML.metrics import rootMeanSquareError


if __name__ == "__main__":

	#fileName = "../datasets/noisy-linear.csv"
	fileName = os.path.join(UML.UMLPath, "datasets/concrete_slump.csv")
	allData = createData("Matrix", fileName, fileType="csv")
	trainX, trainY, testX, testY = splitData(allData, labelID='Compressive Strength', fractionForTestSet=.15)
	#random.seed = 5
	#remove this column, since it's just the ID number of each data point
	trainX.extractFeatures("No")
	testX.extractFeatures("No")



	#do some dimensionality reduction
	normalizeData('mlpy.PCA', trainX, testX=testX, arguments={'k':5})

	"""
	results = runAndTest("mlpy.Ridge", trainX, trainY, testX, testY, arguments={"lmb":1}, performanceFunction=[rootMeanSquareError, meanAbsoluteError])

	print "results", results
	"""


	toRun = 'runAndTest("mlpy.Ridge", trainX, trainY, testX, testY, {"lmb":<.01|.1|1>}, [rootMeanSquareError])'
	runs = functionCombinations(toRun)
	runs.append('runAndTest("mlpy.LARS", trainX, trainY, testX, testY, {"maxsteps":20}, [rootMeanSquareError])')

	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=10, extraParams=locals())


	print "bestFunction:\n", bestFunction
	print "performance: ", performance






	#OLD STUFF

	##results = runAndTest("mlpy.LibSvm", trainX, trainY, testX, testY, arguments={"C":1}, performanceFunction=[fractionIncorrect])

	#xData = createData("Matrix", fileName, fileType="csv")
	#allData = createData("Matrix", fileName, fileType="csv")
	#trainX, trainY, testX, testY = splitData(allData, labelID='y', fractionForTestSet=.15)
	#normalizeData('mlpy.PCA', trainX, testX=testX, arguments={'k':2})



	#split out the label for this data
	#yData = matrix.extractFeatures("Compressive Strength")

	#get rid of feature "No" column since it's just gives each data point a number
	#xData.extractFeatures("No")

	#split into training and testing
