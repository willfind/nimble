from UML.applications.allowImports import boilerplate
boilerplate()
from UML import run
from UML import normalize
from UML import data
from UML import crossValidateReturnBest
from UML import loadTrainingAndTesting
from UML import functionCombinations
from UML import runAndTestDirect
from UML.performance.metric_functions import classificationError


def testDataPrepExample():
	"""
		Functional test for data preparation
	"""

	# string manipulation to get and make paths
	pathOrig = "example_data/adult_income_classification_tiny.csv"
	pathOut = "example_data/adult_income_classification_tiny_numerical.csv"

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
	pathIn = "example_data/adult_income_classification_tiny_numerical.csv"
	trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID='income', fractionForTestSet=.15, loadType="DenseMatrixData", fileType="csv")

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRun = 'runAndTestDirect("mlpy.LibSvm", trainX, testX, trainY, testY, {"C":<.01|.1|.1|10|100>,"gamma":<.01|.1|.1|10|100>,"kernel_type":"<rbf|sigmoid>"}, [classificationError])'
	runs = functionCombinations(toRun)
	extraParams = {'runAndTestDirect':runAndTestDirect, 'classificationError':classificationError}	

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


