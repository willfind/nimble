
from UML import createData
from UML import trainAndTest
from UML.umlHelpers import extractWinningPredictionLabel
from UML.umlHelpers import generateAllPairs
from UML.metrics import fractionIncorrect
from UML.randomness import pythonRandom

def testExtractWinningPredictionLabel():
	"""
	Unit test for extractWinningPrediction function in runner.py
	"""
	predictionData = [[1, 3, 3, 2, 3, 2], [2, 3, 3, 2, 2, 2], [1, 1, 1, 1, 1, 1], [4, 4, 4, 3, 3, 3]]
	BaseObj = createData('Matrix', predictionData)
	BaseObj.transpose()
	predictions = BaseObj.applyToFeatures(extractWinningPredictionLabel, inPlace=False)
	listPredictions = predictions.copyAs(format="python list")
	
	assert listPredictions[0][0] - 3 == 0.0
	assert listPredictions[0][1] - 2 == 0.0
	assert listPredictions[0][2] - 1 == 0.0
	assert (listPredictions[0][3] - 4 == 0.0) or (listPredictions[0][3] - 3 == 0.0)



def testGenerateAllPairs():
	"""
	Unit test function for testGenerateAllPairs
	"""
	testList1 = [1, 2, 3, 4]
	testPairs = generateAllPairs(testList1)
	print testPairs

	assert len(testPairs) == 6
	assert ((1, 2) in testPairs) or ((2, 1) in testPairs)
	assert not (((1, 2) in testPairs) and ((2, 1) in testPairs))
	assert ((1, 3) in testPairs) or ((3, 1) in testPairs)
	assert not (((1, 3) in testPairs) and ((3, 1) in testPairs))
	assert ((1, 4) in testPairs) or ((4, 1) in testPairs)
	assert not (((1, 4) in testPairs) and ((4, 1) in testPairs))
	assert ((2, 3) in testPairs) or ((3, 2) in testPairs)
	assert not (((2, 3) in testPairs) and ((3, 2) in testPairs))
	assert ((2, 4) in testPairs) or ((4, 2) in testPairs)
	assert not (((2, 4) in testPairs) and ((4, 2) in testPairs))
	assert ((3, 4) in testPairs) or ((4, 3) in testPairs)
	assert not (((3, 4) in testPairs) and ((4, 3) in testPairs))

	testList2 = []
	testPairs2 = generateAllPairs(testList2)
	assert testPairs2 is None


#todo set seed and verify that you can regenerate error several times with
#crossValidateReturnBest, trainAndApply, and your own computeMetrics
def test_trainAndTest():
	"""Assert valid results returned for different arguments to the algorithm:
	with default ie no args
	with one argument for the algorithm
	with multiple values for one argument for the algorithm
	with complicated argument for the algorithm
	"""
	variables = ["x1", "x2", "x3", "label"]
	numPoints = 20
	data1 = [[pythonRandom.random(), pythonRandom.random(), pythonRandom.random(), int(pythonRandom.random()*3)+1] for _pt in xrange(numPoints)]
	# data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	trainObj1 = createData('Matrix', data=data1, featureNames=variables)

	testData1 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3]]
	testObj1 = createData('Matrix', data=testData1)

	#with default ie no args
	runError = trainAndTest('Custom.KNNClassifier', trainObj1, 3, testObj1, 3, fractionIncorrect)
	assert isinstance(runError, float)

	#with one argument for the algorithm
	runError = trainAndTest('Custom.KNNClassifier', trainObj1, 3, testObj1, 3, fractionIncorrect, k=1)
	assert isinstance(runError, float)

	#with multiple values for one argument for the algorithm
	runError = trainAndTest('Custom.KNNClassifier', trainObj1, 3, testObj1, 3, fractionIncorrect, k=(1,2))
	assert isinstance(runError, float)

	#with complicated argument for the algorithm
#	runError = trainAndTest('Custom.KNNClassifier', trainObj1, 3, testObj1, 3, fractionIncorrect, k=(1,2), p=(1,2))
#	assert isinstance(runError, float)


