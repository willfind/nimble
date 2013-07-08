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

	pathIn = "datasets/adult_income_classification_tiny_numerical.csv"
	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID="income", fractionForTestSet=.2, loadType="Sparse", fileType="csv")
	print "Finished loading data"

	# sparse types aren't playing nice with the error metrics currently, so convert
	trainY = trainY.toDense()
	testY = testY.toDense()

	args = {"kernel":"GaussianKernel", "width":1, "C":1}
	results = runAndTest("shogun.MulticlassLibSVM", trainX, testX, trainY, testY, args, [classificationError])
	rawResults = run("shogun.MulticlassLibSVM", trainX, testX, dependentVar=trainY, arguments=args)
	
	print "results: "+repr(results)
	print "raw predictions: "+repr(rawResults.data)
