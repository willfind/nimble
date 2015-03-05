
from UML import createData
from UML import trainAndTest
from UML.helpers import extractWinningPredictionLabel
from UML.helpers import generateAllPairs
from UML.calculate import fractionIncorrect
from UML.randomness import pythonRandom




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




