#tests for functions relating to ArgumentIterator class
#which is used by
#crossValidateReturnBest
#crossValidateReturnAll
#trainAndTest
import sys
sys.path.append('../..')

from UML.helpers import ArgumentIterator
from UML.helpers import _buildArgPermutationsList

	# example call to _buildArgPermutationsList:
	# if rawArgInput is {'a':(1,2,3), 'b':(4,5)}
	# then _buildArgPermutationsList([],{},0,rawArgInput)
	# returns [{'a':1, 'b':4}, {'a':2, 'b':4}, {'a':3, 'b':4}, {'a':1, 'b':5}, {'a':2, 'b':5}, {'a':3, 'b':5},]

def test_buildArgPermutationsList():
	"""Assert that the permutations are exhaustive"""

	argumentDict = {'a':(1,2,3), 'b':(4,5)}
	returned = _buildArgPermutationsList([], {}, 0, argumentDict)
	assert returned
	tupleLengthsList = []
	for curTuple in argumentDict.values():
		try:
			tupleLengthsList.append(len(curTuple))
		except TypeError: #has length 1
			tupleLengthsList.append(1)

	numberOfDistinctArgumentCombinations = reduce(lambda x,y: x*y, tupleLengthsList)
	assert len(returned) == numberOfDistinctArgumentCombinations


	#do a hardcorded example
	returned = _buildArgPermutationsList([], {}, 0, {'a':(1,2,3), 'b':(4,5)})
	shouldBeList = [{'a':1, 'b':4}, {'a':2, 'b':4}, {'a':3, 'b':4}, {'a':1, 'b':5}, {'a':2, 'b':5}, {'a':3, 'b':5}]
	shouldBeListOfStrings = [str(curHash) for curHash in shouldBeList]
	returnedListOfStrings = [str(curHash) for curHash in returned]
	assert set(shouldBeListOfStrings) == set(returnedListOfStrings)


def test_ArgumentIterator():
	"""Assert that argument iterator can handle empty arguments and
	iterates as otherwise expected via _buildArgPermutationsList"""
	#assert works with empty dict
	returned = ArgumentIterator({})
	assert {} == returned.next()
	#should be out of iterations after popping the empty list
	try:
		returned.next()
		#if next works a second time, then ArgumentIterator has too many iterations
		assert False
	except StopIteration:
		pass

	argumentDict = {'a':(1,2,3), 'b':(4,5)}
	returned = ArgumentIterator(argumentDict)

	iterationCount = 0
	for curArgumentCombo in returned:
		iterationCount += 1
		assert set(argumentDict.keys()) == set(curArgumentCombo.keys())
		assert len(argumentDict.keys()) == len(curArgumentCombo.keys())

	assert iterationCount == len(_buildArgPermutationsList([], {}, 0, argumentDict))



def test_ArgumentIterator_stringsAndTuples():
	arguments = {'a':'hello', 'b':(1,2,5)}

	returned = ArgumentIterator(arguments)

	for curr in returned:
		assert curr['a'] == 'hello'
		assert curr['b'] in (1,2,5)

		assert len(curr.keys()) == 2
