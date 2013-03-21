"""
Script that loads the preprocessed adult income dataset as a sparse matrix,
then runs a trial with a shogun svm classisifer with kernel.

"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	from UML import runAndTest
	from UML import loadTrainingAndTesting
	from UML.performance.metric_functions import classificationError

	pathIn = "example_data/adult_income_classification_tiny_numerical.csv"
	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID='income', fractionForTestSet=.2, loadType="CooSparseData", fileType="csv")

	# sparse types aren't playing nice with the error metrics currently, so convert
	trainY = trainY.toDenseMatrixData()
	testY = testY.toDenseMatrixData()

	args = {"kernel":"GaussianKernel", "width":1, "C":1}
	results = runAndTest("shogun.LibSVMMultiClass", trainX, testX, trainY, testY, args, [classificationError])
	
	print results
