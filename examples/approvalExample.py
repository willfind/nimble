"""
Script that loads the preprocessed adult income dataset as a sparse matrix,
then runs a trial with a shogun svm classisifer with kernel.

"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	import os.path
	import UML
	from UML import runAndTest
	from UML import run
	from UML import createData
	from UML import splitData
	from UML.metrics import fractionIncorrect

	pathIn = os.path.join(UML.UMLPath, "datasets/sparseSample.mtx")
	allData = createData("Sparse", pathIn, fileType="mtx")
	trainX, trainY, testX, testY = splitData(allData, labelID=5, fractionForTestSet=.2)

	# sparse types aren't playing nice with the error metrics currently, so convert
	trainY = trainY.toMatrix()
	testY = testY.toMatrix()

	args = {"kernel":"GaussianKernel", "C":1}
	results = runAndTest("shogun.MulticlassLibSVM", trainX.duplicate(), testX.duplicate(), trainY.duplicate(), testY.duplicate(), args, [fractionIncorrect])
	rawResults = run("shogun.MulticlassLibSVM", trainX, testX, trainY, args)
	
	print results
	print str(rawResults.data)
