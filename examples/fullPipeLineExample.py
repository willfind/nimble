"""
Short module demonstrating the full pipeline of train - test - log results.
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	from UML import createData
	from UML import runAndTest
	from UML.runners import runOneVsOne
	from UML.runners import runAndTestOneVsOne
	from UML.metrics import fractionIncorrect

	variables = ["x1","x2","x3", "label"]
	data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	trainObj = createData('Matrix', data1, variables)

	data2 = [[1,0,0,1],[0,1,0,2],[0,0,1,3]]
	testObj = createData('Matrix', data2, variables)

	trainObj2 = trainObj.copy()
	testObj2 = testObj.copy()

	trainObj3 = trainObj.copy()
	testObj3 = testObj.copy()

	metricFuncs = []
	metricFuncs.append(fractionIncorrect)

	results1 = runAndTest('sciKitLearn.LogisticRegression',trainObj, trainY=3, testX=testObj, testY=3, arguments={}, performanceMetricFuncs=metricFuncs)
	results2 = runOneVsOne('sciKitLearn.SVC',trainObj, testObj, trainDependentVar=3, testDependentVar=3, arguments={}, scoreMode='label', sendToLog=False)
	results3 = runAndTestOneVsOne('sciKitLearn.SVC',trainObj, testObj, trainDependentVar=3, testDependentVar=3, arguments={}, performanceMetricFuncs=metricFuncs, sendToLog=False)
	resultsBestScore = runOneVsOne('sciKitLearn.SVC',trainObj2, testObj2, trainDependentVar=3, testDependentVar=3, arguments={}, scoreMode='bestScore', sendToLog=False)
	resultsAllScores = runOneVsOne('sciKitLearn.SVC',trainObj3, testObj2, trainDependentVar=3, testDependentVar=3, arguments={}, scoreMode='allScores', sendToLog=False)

	print 'Standard run results: '+str(results1)
	print 'One vs One predictions: '+repr(results2.data)
	print 'One vs One, best score, column headers: ' + repr(resultsBestScore.featureNames)
	print 'One vs One best score: '+repr(resultsBestScore.data)
	print 'One vs One, all scores, column headers: ' + repr(resultsAllScores.featureNames)
	print 'One vs One all scores: '+repr(resultsAllScores.data)
	print 'One vs One performance results: ' + str(results3)
