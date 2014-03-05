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
	from UML import splitData
	from UML.metrics import fractionIncorrect

	pathIn = os.path.join(UML.UMLPath, "datasets/sparseSample.mtx")
	allData = createData("Sparse", pathIn, fileType="mtx")
	trainX, trainY, testX, testY = splitData(allData, labelID=5, fractionForTestSet=.2)

	# sparse types aren't playing nice with the error metrics currently, so convert
	trainY = trainY.copyAs(format="Matrix")
	testY = testY.copyAs(format="Matrix")

	args = {"kernel":"GaussianKernel", "C":1}
	results = trainAndTest("shogun.MulticlassLibSVM", trainX.copy(), trainY.copy(), testX.copy(), testY.copy(), args, [fractionIncorrect])
	rawResults = trainAndApply("shogun.MulticlassLibSVM", trainX, trainY, testX, args)
	
	print results
	print str(rawResults.data)
