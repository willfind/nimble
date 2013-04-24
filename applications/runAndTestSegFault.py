"""
Script that uses 50K points of job posts to try to predict approved/rejected status
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	from UML import crossValidateReturnBest
	from UML import functionCombinations
	from UML.combinations.Combinations import executeCode
	from UML import runAndTest
	from UML import data
	from UML import loadTrainingAndTesting
	from UML.performance.metric_functions import classificationError

	pathIn = "applications/example_data/sparseSampleReal.mtx"
	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID=0, fractionForTestSet=.2, loadType="CooSparseData", fileType="mtx")

	# sparse types aren't playing nice with the error metrics currently, so convert
	trainY = trainY.toDenseMatrixData()
	testY = testY.toDenseMatrixData()

	trainYList = []
	
	for i in range(len(trainY.data)):
		label = trainY.data[i][0]
		trainYList.append([int(label)])

	testYList = []
	for i in range(len(testY.data)):
		label = testY.data[i][0]
		testYList.append([int(label)])

	trainY = data('dense', trainYList)
	testY = data('dense', testYList)

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRun = 'runAndTest("shogun.MulticlassOCAS", trainX, testX, trainY, testY, {"C":<1.0>}, [classificationError])'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTest':runAndTest, 'classificationError':classificationError}
	results = {}
	for run in runs:
		dataHash={"trainX": trainX.duplicate(), 
		          "testX":testX.duplicate(), 
		          "trainY":trainY.duplicate(), 
		          "testY":testY.duplicate(), 
		          'runAndTest':runAndTest, 
		          'classificationError':classificationError}
		print "Run call: "+repr(run)
		print "Run results: "+repr(executeCode(run, dataHash))