"""
Script that loads the preprocessed adult income dataset as a sparse matrix,
then runs a trial with a shogun svm classisifer with kernel.

"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	import os.path
	import UML
	from UML import trainAndTest
	from UML import trainAndApply
	from UML import createData
	from UML.calculate import fractionIncorrect

	pathIn = os.path.join(UML.UMLPath, "datasets/adult_income_classification_tiny_numerical.csv")
	allData = createData("Sparse", pathIn, fileType="csv")
	trainX, trainY, testX, testY = allData.trainAndTestSets(testFraction=.2, labels="income")
	print "Finished loading data"

	# sparse types aren't playing nice with the error metrics currently, so convert
	trainY = trainY.copyAs(format="Matrix")
	testY = testY.copyAs(format="Matrix")

	args = {"kernel":"GaussianKernel", "width":1, "C":1}
	results = trainAndTest("shogun.MulticlassLibSVM", trainX, trainY, testX, testY, args, [fractionIncorrect])
	rawResults = trainAndApply("shogun.MulticlassLibSVM", trainX, trainY, testX, arguments=args)
	
	print "results: "+repr(results)
	print "raw predictions: "+repr(rawResults.data)
