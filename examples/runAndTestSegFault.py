"""
Script that uses 50K points of job posts to try to predict approved/rejected status
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
	import os.path
	import UML
	from UML import functionCombinations
	from UML.umlHelpers import executeCode
	from UML import runAndTest
	from UML import createData
	from UML import splitData
	from UML.metrics import fractionIncorrect

	pathIn = os.path.join(UML.UMLPath, "datasets/sparseSampleReal.mtx")
	allData = createData("Sparse", pathIn, fileType="mtx")
	trainX, trainY, testX, testY = splitData(allData, labelID=0, fractionForTestSet=.2)

	# sparse types aren't playing nice with the error metrics currently, so convert
	trainY = trainY.copyAs(format="Matrix")
	testY = testY.copyAs(format="Matrix")

	trainYList = []
	
	for i in range(len(trainY.data)):
		label = trainY.data[i][0]
		trainYList.append([int(label)])
		print "label: "+str(int(label))

	testYList = []
	for i in range(len(testY.data)):
		label = testY.data[i][0]
		testYList.append([int(label)])
		print "label: "+str(int(label))

	trainY = createData('Matrix', trainYList)
	testY = createData('Matrix', testYList)

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRun = 'runAndTest("shogun.MulticlassOCAS", trainX, trainY, testX, testY, {"C":<1.0>}, [fractionIncorrect])'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTest':runAndTest, 'fractionIncorrect':fractionIncorrect}
	results = {}
	for run in runs:
		dataHash={"trainX": trainX.copy(), 
		          "testX":testX.copy(), 
		          "trainY":trainY.copy(), 
		          "testY":testY.copy(), 
		          'runAndTest':runAndTest, 
		          'fractionIncorrect':fractionIncorrect}
		print "Run call: "+repr(run)
		print "Run results: "+repr(executeCode(run, dataHash))