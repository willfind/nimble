"""
Short module demonstrating the full pipeline of train - test - log results.
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	from UML import createData
	from UML import trainAndTest
	from UML import trainAndApply
	from UML.calculate import fractionIncorrect

	variables = ["x1","x2","x3", "label"]
	data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	trainObj = createData('Matrix', data=data1, featureNames=variables)

	data2 = [[1,0,0,1],[0,1,0,2],[0,0,1,3]]
	testObj = createData('Matrix', data=data2, featureNames=variables)

	trainObj2 = trainObj.copy()
	testObj2 = testObj.copy()

	trainObj3 = trainObj.copy()
	testObj3 = testObj.copy()

	results1 = trainAndTest('sciKitLearn.LogisticRegression',trainObj, trainY=3, testX=testObj, testY=3, arguments={}, performanceFunction=fractionIncorrect, useLog=False)
	results2 = trainAndApply('sciKitLearn.SVC', trainObj, trainY=3, testX=testObj, arguments={}, scoreMode='label', multiClassStrategy="OneVsOne", useLog=False)
#	results3 = trainAndTest('sciKitLearn.SVC',trainObj, trainY=3, testX=testObj, testY=3, arguments={}, performanceFunction=fractionIncorrect, useLog=False)
	resultsBestScore = trainAndApply('sciKitLearn.SVC',trainObj2, trainY=3, testX=testObj2, arguments={}, scoreMode='bestScore', multiClassStrategy="OneVsOne", useLog=False)
	resultsAllScores = trainAndApply('sciKitLearn.SVC',trainObj3, trainY=3, testX=testObj2, arguments={}, scoreMode='allScores', multiClassStrategy="OneVsOne", useLog=False)

	print 'Standard run results: '+str(results1)
	print 'One vs One predictions: '+repr(results2.data)
	print 'One vs One, best score, column headers: ' + repr(resultsBestScore.featureNames)
	print 'One vs One best score: '+repr(resultsBestScore.data)
	print 'One vs One, all scores, column headers: ' + repr(resultsAllScores.featureNames)
	print 'One vs One all scores: '+repr(resultsAllScores.data)
#	print 'One vs One performance results: ' + str(results3)
