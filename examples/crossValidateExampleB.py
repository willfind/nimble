

from allowImports import boilerplate
boilerplate()


if __name__ == "__main__":

	from UML import crossValidateReturnBest
	from UML import createData
	from UML import splitData
	from UML import functionCombinations
	from UML.metrics import classificationError

	# path to input specified by command line argument
	pathIn = "../datasets/adult_income_classification_tiny_numerical.csv"
	allData = createData("Matrix",pathIn, fileType="csv")
	trainX, trainY, testX, testY = splitData(allData, labelID='income', fractionForTestSet=.15)

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRun = 'runAndTestDirect("mlpy.LibSvm", trainX, testX, trainY, testY, {"C":<.01|.1|1>,"gamma":<.01|.1|1>,"kernel_type":"<rbf|sigmoid>"}, [classificationError])'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTestDirect':runAndTestDirect, 'classificationError':classificationError}

	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=10, extraParams=extraParams)
	print bestFunction
	print performance

