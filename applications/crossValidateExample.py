

from allowImports import boilerplate
boilerplate()


if __name__ == "__main__":

	import sys

	from UML import crossValidate
	from UML import crossValidateReturnBest
	from UML import loadTrainingAndTesting
	from UML import functionCombinations
	from UML import normalize

	# path to input specified by command line argument
	pathIn = sys.argv[1]

	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID=0, fractionForTestSet=.15, loadType="DenseMatrixData", fileType="csv")

	#we'll be trying all combinations of C in [0.1, 1, 10, 100] and iterations in [100, 1000]
	runs = functionCombinations('from UML import runWithClassificationError;runWithClassificationError("mlpy", "KNN", trainX, trainY, testX, testY, arguments={"k":<1|5|10|50|100>})')

	#this will return the text of whichever function performed better, as well as that best performance value
	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, minimize=False, numFolds=10)
	print bestFunction
	print performance

