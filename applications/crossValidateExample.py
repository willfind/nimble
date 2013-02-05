

from allowImports import boilerplate
boilerplate()


if __name__ == "__main__":

	import sys

	from UML import crossValidateReturnBest
	from UML import loadTrainingAndTesting
	from UML import functionCombinations
	from UML import runAndTestDirect
	from UML import runAndTest
	from UML import run
	from UML.performance.metric_functions import classificationError

	# path to input specified by command line argument
	pathIn = sys.argv[1]
	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID=0, fractionForTestSet=.15, loadType="DenseMatrixData", fileType="csv")

	# run and test with a direct call to run()
	toRun = 'runAndTestDirect("mlpy.KNN", trainX, testX, trainY, testY, {"k":<1|5|10|50|100>}, [classificationError])'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTestDirect':runAndTestDirect, 'classificationError':classificationError}

	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, mode='max', numFolds=10, extraParams=extraParams)
	print bestFunction
	print performance

	# run and test with a constructed call to run()
	runCall = '"run(\'mlpy.KNN\', trainX, testX, dependentVar=dependentVar, arguments={\'k\':<1|5|10|50|100>})"'
	toRun = 'runAndTest(trainX, testX, trainY, testY,' + runCall +',[classificationError])'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTest':runAndTest, 'classificationError':classificationError, 'run':run}

	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, mode='max', numFolds=10, extraParams=extraParams)
	print bestFunction
	print performance

