

from allowImports import boilerplate
boilerplate()


if __name__ == "__main__":

	from UML import crossValidateReturnBest
	from UML import loadTrainingAndTesting
	from UML import functionCombinations
	from UML import runAndTestDirect
	from UML.performance.metric_functions import classificationError

	# path to input specified by command line argument
	pathIn = "example_data/adult_income_classification_tiny.csv"
	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID='income', fractionForTestSet=.15, loadType="DenseMatrixData", fileType="csv")

	# run and test with a direct call to run()
	toRun = 'runAndTestDirect("mlpy.KNN", trainX, testX, trainY, testY, {"k":<1|5|10|15>}, [classificationError])'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTestDirect':runAndTestDirect, 'classificationError':classificationError}

	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=10, extraParams=extraParams)
	print bestFunction
	print performance

