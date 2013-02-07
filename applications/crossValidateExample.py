

from allowImports import boilerplate
boilerplate()


if __name__ == "__main__":

	from UML import crossValidateReturnBest
	from UML import loadTrainingAndTesting
	from UML import functionCombinations
	from UML import runAndTestDirect
	from UML import runAndTest
	from UML import run
	from UML.performance.metric_functions import classificationError

	# path to input specified by command line argument
	pathIn = "example_data/adult_income_classification_tiny.csv"
	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID='income', fractionForTestSet=.15, loadType="DenseMatrixData", fileType="csv")

	# run and test with a direct call to run()
	toRun = 'runAndTestDirect("sciKitLearn.LogisticRegression", trainX, testX, trainY, testY, {"C":<1|.05|.01|.005|.001>}, [classificationError])'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTestDirect':runAndTestDirect, 'classificationError':classificationError}

	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=10, extraParams=extraParams)
	print bestFunction
	print performance

	# run and test with a constructed call to run()
	runCall = '"run(\'sciKitLearn.LogisticRegression\', trainX, testX, dependentVar=dependentVar, arguments={\'C\':<1|.05|.01|.005|.001>})"'
	toRun = 'runAndTest(trainX, testX, trainY, testY,' + runCall +',[classificationError])'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTest':runAndTest, 'classificationError':classificationError, 'run':run}

	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=10, extraParams=extraParams)
	print bestFunction
	print performance

