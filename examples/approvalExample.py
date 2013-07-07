"""
Script that loads the preprocessed adult income dataset as a sparse matrix,
then runs a trial with a shogun svm classisifer with kernel.

"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	from UML import runAndTest
	from UML import run
	from UML import loadTrainingAndTesting
	from UML.metrics import classificationError

	pathIn = "datasets/sparseSample.mtx"
	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID=5, fractionForTestSet=.2, loadType="Sparse", fileType="mtx")

	# sparse types aren't playing nice with the error metrics currently, so convert
	trainY = trainY.toDenseMatrixData()
	testY = testY.toDenseMatrixData()

	args = {"kernel":"GaussianKernel", "C":1}
	results = runAndTest("shogun.MulticlassLibSVM", trainX.duplicate(), testX.duplicate(), trainY.duplicate(), testY.duplicate(), args, [classificationError])
	rawResults = run("shogun.MulticlassLibSVM", trainX, testX, trainY, testY, args)
	
	print results
	print str(rawResults.data)
