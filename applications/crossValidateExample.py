

from allowImports import boilerplate
boilerplate()


if __name__ == "__main__":

	import sys

	from UML import crossValidate
	from UML import crossValidateReturnBest
	from UML import loadTrainingAndTesting
	from UML import functionCombinations
	from UML import normalize
	from UML import runWithClassificationError

	# path to input specified by command line argument
	pathIn = sys.argv[1]

	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID=0, fractionForTestSet=.15, loadType="DenseMatrixData", fileType="csv")
	runs = functionCombinations('runWithClassificationError("mlpy", "KNN", trainX, trainY, testX, testY, arguments={"k":<1|5|10|50|100>})')
	extraParams = {'runWithClassificationError':runWithClassificationError}

	#this will return the text of whichever function performed better, as well as that best performance value
	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, mode='max', numFolds=10, extraParams=extraParams)
	print bestFunction
	print performance

