"""
Script that uses 50K points of job posts to try to predict approved/rejected status
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	from UML import crossValidateReturnBest
	from UML import functionCombinations
	from UML.umlHelpers import executeCode
	from UML import runAndTest
	from UML import create
	from UML import loadTrainingAndTesting
	from UML.metrics import proportionPercentNegative90

	pathIn = "/media/library_/LaddersData/PlaygroundFull/DocVectors.mtx"
	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID=0, fractionForTestSet=.2, loadType="CooSparseData", fileType="mtx")
	print "Finished loading data"
	print "trainX shape: " + str(trainX.data.shape)
	print "trainY shape: " + str(trainY.data.shape)

	# sparse types aren't playing nice with the error metrics currently, so convert
	trainY = trainY.toDenseMatrixData()
	testY = testY.toDenseMatrixData()

	trainYList = []
	
	for i in range(len(trainY.data)):
		label = trainY.data[i][0]
		if label == '1' or label == '2' or label == 1 or label == 2:
			trainYList.append([int(label)])
		else:
			trainYList.append([2])

	testYList = []
	for i in range(len(testY.data)):
		label = testY.data[i][0]
		if label == '1' or label == '2' or label == 1 or label == 2:
			testYList.append([int(label)])
		else:
			testYList.append([2])

	trainY = create('dense', trainYList)
	testY = create('dense', testYList)

	print "Finished converting labels to ints"

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRun = 'runAndTest("shogun.MulticlassLibLinear", trainX, testX, trainY, testY, {"C":<0.01|0.1|0.5|1.0|10.0>}, [proportionPercentNegative90], scoreMode="allScores", negativeLabel="2", sendToLog=False)'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTest':runAndTest, 'proportionPercentNegative90':proportionPercentNegative90}
	results = {}
	run, results = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=5, extraParams=extraParams, sendToLog=True)

	# for run in runs:
	run = run.replace('sendToLog=False', 'sendToLog=True')
	dataHash={"trainX": trainX.duplicate(), 
		      "testX":testX.duplicate(), 
		      "trainY":trainY.duplicate(), 
		      "testY":testY.duplicate(), 
		      'runAndTest':runAndTest, 
		      'proportionPercentNegative90':proportionPercentNegative90}
	# 	print "Run call: "+repr(run)
	print "Best Run confirmation: "+repr(executeCode(run, dataHash))


