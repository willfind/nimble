from UML.examples.allowImports import boilerplate
boilerplate()
from UML import run
from UML import normalize
from UML import data
from UML import crossValidateReturnBest
from UML import crossValidate
from UML import loadTrainingAndTesting
from UML import functionCombinations
from UML import runAndTest
from UML.metrics import classificationError

import os
exampleDirPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/datasets/"

def testEverythingVolumeOne():
	"""
	Try to test some full use cases: load data, split data, normalize data, run crossValidate or
	crossValidateReturnBest, and get results.  Use the classic iris data set for classification.
	"""
	pathOrig = os.path.join(os.path.dirname(__file__), "../../datasets/iris.csv")

	# we specify that we want a DenseMatrixData object returned, and with just the path it will
	# decide automaticallly the format of the file that is being loaded
	processed = data("DenseMatrixData", pathOrig)

	assert processed.data is not None

	partOne = processed.extractPointsByCoinToss(0.5)
	partOneTest = partOne.extractPointsByCoinToss(0.1)
	partTwoX = processed
	partTwoY = processed.extractFeatures('Type')

	assert partOne.points() > 55
	assert partOne.points() < 80
	assert partTwoX.points() > 65
	assert partTwoX.points() < 85
	assert partTwoY.points() == partTwoX.points()
	assert partOne.points() + partTwoX.points() + partOneTest.points() == 150

	trainX = partOne
	trainY = partOne.extractFeatures('Type')
	testX = partOneTest
	testY = partOneTest.extractFeatures('Type')
	

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRunOne = 'runAndTest("mlpy.LibSvm", trainX, testX, trainY, testY, {"C":<.01|.1|.1|10|100>,"gamma":<.01|.1|.1|10|100>,"kernel_type":"<rbf|sigmoid>"}, [classificationError])'
	runsOne = functionCombinations(toRunOne)
	extraParams = {'runAndTest':runAndTest, 'classificationError':classificationError}
	fullCrossValidateResults = crossValidate(trainX, trainY, runsOne, numFolds=10, extraParams=extraParams, sendToLog=False)
	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runsOne, mode='min', numFolds=10, extraParams=extraParams, sendToLog=False)

	#Check that the error rate for each function is between 0 and 1
	for result in fullCrossValidateResults.items():
		assert result[1] >= 0.0
		assert result[1] <= 1.0
	assert bestFunction is not None
	assert performance >= 0.0
	assert performance <= 1.0

	trainObj = trainX
	testObj = partTwoX

	# use normalize to modify our data; we call a dimentionality reduction algorithm to
	# simply our mostly redundant points. k is the desired number of dimensions in the output
	normalize('mlpy.PCA', trainObj, testObj, arguments={'k':1})

	# assert that we actually do have fewer dimensions
	assert trainObj.data[0].size == 1
	assert testObj.data[0].size == 1

def testDataPrepExample():
	"""
		Functional test for data preparation
	"""

	# string manipulation to get and make paths
	pathOrig = os.path.join(os.path.dirname(__file__), "../../datasets/adult_income_classification_tiny.csv")
	pathOut = os.path.join(os.path.dirname(__file__), "../../datasets/adult_income_classification_tiny_numerical.csv")

	# we specify that we want a DenseMatrixData object returned, and with just the path it will
	# decide automaticallly the format of the file that is being loaded
	processed = data("RowListData", pathOrig)

	# this feature is a precalculated similarity rating. Lets not make it too easy....
	processed.extractFeatures('fnlwgt')

	#convert assorted features from strings to binary category columns
	processed.featureToBinaryCategoryFeatures('sex')
	processed.featureToBinaryCategoryFeatures('marital-status')
	processed.featureToBinaryCategoryFeatures('occupation')
	processed.featureToBinaryCategoryFeatures('relationship')
	processed.featureToBinaryCategoryFeatures('race')
	processed.featureToBinaryCategoryFeatures('native-country')

	# convert 'income' column (the classification label) to a single numerical column
	processed.featureToIntegerCategories('income')

	#scrub the rest of the string valued data -- the ones we converted are the non-redundent ones
	processed.dropStringValuedFeatures()

	# output the split and normalized sets for later usage
	processed.writeFile('csv', pathOut, includeFeatureNames=True)

def testCrossValidateExample():
	"""
		Functional test for load-data-to-classification-results example of crossvalidation
	"""
	# path to input specified by command line argument
	pathIn = os.path.join(os.path.dirname(__file__), "../../datasets/adult_income_classification_tiny_numerical.csv")
	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID='income', fractionForTestSet=.15, loadType="DenseMatrixData", fileType="csv")

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRun = 'runAndTest("mlpy.LibSvm", trainX, testX, trainY, testY, {"C":<.01|.1|.1|10|100>,"gamma":<.01|.1|.1|10|100>,"kernel_type":"<rbf|sigmoid>"}, [classificationError])'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTest':runAndTest, 'classificationError':classificationError}

	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=10, extraParams=extraParams)
	assert bestFunction is not None
	assert performance > 0.0

def testNormalizing():
	"""
		Functional test of data normalization
	"""
	# we separate into classes accoring to whether x1 is positive or negative
	variables = ["y","x1","x2","x3"]
	data1 = [[1,6,0,0], [1,3,0,0], [0,-5,0,0],[0,-3,0,0]]
	trainObj = data('DenseMatrixData', data1, variables)
	trainObjY = trainObj.extractFeatures('y')

	# data we're going to classify
	data2 = [[1,0,0],[4,0,0],[-1,0,0], [-2,0,0]]
	testObj = data('DenseMatrixData', data2)

	# baseline check
	assert trainObj.data[0].size == 3
	assert testObj.data[0].size == 3

	# use normalize to modify our data; we call a dimentionality reduction algorithm to
	# simply our mostly redundant points. k is the desired number of dimensions in the output
	normalize('mlpy.PCA', trainObj, testObj, arguments={'k':1})

	# assert that we actually do have fewer dimensions
	assert trainObj.data[0].size == 1
	assert testObj.data[0].size == 1

	ret = run('mlpy.KNN', trainObj, testObj, dependentVar=trainObjY, arguments={'k':1})

	# assert we get the correct classes
	assert ret.data[0,0] == 1
	assert ret.data[1,0] == 1
	assert ret.data[2,0] == 0
	assert ret.data[3,0] == 0


