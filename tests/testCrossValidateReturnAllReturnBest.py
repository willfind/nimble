#tests for 
#crossValidateReturnAll
#crossValidateReturnBest

#so you can run as main:
import sys
sys.path.append('../..')


from UML import crossValidateReturnAll
from UML import crossValidateReturnBest
from UML import createData
from UML import createRandomData
from UML.metrics import *
import random
from pdb import set_trace as ttt



def _randomLabeledDataSet(dataType='matrix', numPoints=100, numFeatures=5, numLabels=3, seed=None):
	"""returns a tuple of two data objects of type dataType
	the first object in the tuple contains the feature information ('X' in UML language)
	the second object in the tuple contains the labels for each feature ('Y' in UML language)
	"""
	random.seed(seed)
	if numLabels is None:
		labelsRaw = [[random.random()] for _x in xrange(numPoints)]
	else: #labels data set
		labelsRaw = [[int(random.random()*numLabels)] for _x in xrange(numPoints)]

	rawFeatures = [[random.random() for _x in xrange(numFeatures)] for _y in xrange(numPoints)]

	return (createData(dataType, rawFeatures), createData(dataType, labelsRaw))



def test_crossValidateReturnAll():
	"""assert that KNeighborsClassifier generates results with default arguments
	assert that having the same function arguments yields the same results.
	assert that return all gives a cross validated performance for all of its 
	parameter permutations
	"""
	X, Y = _randomLabeledDataSet(numPoints=50, numFeatures=10, numLabels=5)
	#try with no extra arguments at all:
	result = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect, foldSeed='myseed', )
	print result
	assert result
	assert 1 == len(result)
	assert result[0][0] == {}
	#try with some extra elements but all after default
	result = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect, k=(1,2,3))
	print result
	assert result
	assert 3 == len(result)


	#since the same seed is used, and these calls are effectively building the same arguments, (p=2 is default for algo)
	#the scores in results list should be the same (though the keys will be different (one the second will have 'p':2 in the keys as well))
	resultDifferentNeighbors = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect, foldSeed='myseed', k=(1,2,3,4,5))
	resultDifferentNeighborsButSameCombinations = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect, foldSeed='myseed', k=(1,2,3,4,5))
	#assert the the resulting SCORES are identical
	#uncertain about the order
	resultOneScores = [curEntry[1] for curEntry in resultDifferentNeighbors]
	resultTwoScores = [curEntry[1] for curEntry in resultDifferentNeighborsButSameCombinations]
	resultsOneSet = set(resultOneScores)
	resultsTwoSet = set(resultTwoScores)
	assert resultsOneSet == resultsTwoSet

	#assert results have the expected data structure:
	#a list of tuples where the first entry is the argument dict
	#and second entry is the score (float)
	assert isinstance(resultDifferentNeighbors, list)
	for curResult in resultDifferentNeighbors:
		assert isinstance(curResult, tuple)
		assert isinstance(curResult[0], dict)
		assert isinstance(curResult[1], float)





def test_crossValidateReturnBest():
	"""test that the 'best' ie fittest argument combination is chosen.
	test that best tuple is in the 'all' list of tuples.
	"""
	#assert that it returns the best, enforce a seed?
	X, Y = _randomLabeledDataSet(numPoints=50, numFeatures=10, numLabels=5)
	#try with no extra arguments at all:
	shouldMaximizeScores = False

	resultTuple = crossValidateReturnBest('Custom.KNNClassifier', X, Y, fractionIncorrect, foldSeed='myseed', maximize=shouldMaximizeScores, k=(1,2,3))
	assert resultTuple

	allResultsList = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect, foldSeed='myseed', k=(1,2,3))
	#since same args were used (except return all doesn't have a 'maximize' parameter,
	# the best tuple should be in allResultsList
	allArguments = [curResult[0] for curResult in allResultsList]
	allScores = [curResult[1] for curResult in allResultsList]
	assert resultTuple[0] in allArguments
	assert resultTuple[1] in allScores

	#crudely verify that resultTuple was in fact the best in allResultsList
	for curError in allScores:
		#assert that the error is not 'better' than our best error:
		if shouldMaximizeScores:
			assert curError <= resultTuple[1]
		else:
			assert curError >= resultTuple[1]

def test_crossValidateReturnEtc_withDefaultArgs():
	"""Assert that return best and return all work with default arguments as predicted
	ie generating scores for '{}' as the arguments
	"""
	X, Y = _randomLabeledDataSet(numPoints=50, numFeatures=10, numLabels=5)
	#run with default arguments
	bestTuple = crossValidateReturnBest('Custom.KNNClassifier', X, Y, fractionIncorrect, )
	assert bestTuple
	assert isinstance(bestTuple, tuple)
	assert bestTuple[0] == {}
	#run return all with default arguments
	allResultsList = crossValidateReturnAll('Custom.KNNClassifier', X, Y, fractionIncorrect, )
	assert allResultsList
	assert 1 == len(allResultsList)
	assert allResultsList[0][0] == {}

def main():
	test_crossValidateReturnAll()
	test_crossValidateReturnBest()
	test_crossValidateReturnEtc_withDefaultArgs()


if __name__ == '__main__':
	main()
