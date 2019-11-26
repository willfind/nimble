"""
tests for functions relating to ArgumentIterator class
"""

import sys
import copy
from functools import reduce

sys.path.append('../..')

from nimble import CV
from nimble.helpers import ArgumentIterator

# example call to _buildArgPermutationsList:
# if rawArgInput is {'a':(1,2,3), 'b':(4,5)}
# then _buildArgPermutationsList([],{},0,rawArgInput)
# returns [{'a':1, 'b':4}, {'a':2, 'b':4}, {'a':3, 'b':4}, {'a':1, 'b':5}, {'a':2, 'b':5}, {'a':3, 'b':5},]

def test_argPermutationsList():
    """Assert that the permutations are exhaustive"""

    argumentDict = {'a': CV([1, 2, 3]), 'b': CV([4, 5])}
    argIter = ArgumentIterator(argumentDict)
    returned = argIter.permutationsList
    assert returned
    tupleLengthsList = []
    for curTuple in argumentDict.values():
        try:
            tupleLengthsList.append(len(curTuple))
        except TypeError: #has length 1
            tupleLengthsList.append(1)

    numberOfDistinctArgumentCombinations = reduce(lambda x, y: x * y, tupleLengthsList)
    assert len(returned) == numberOfDistinctArgumentCombinations


    #do a hardcorded example
    shouldBeList = [{'a': 1, 'b': 4}, {'a': 2, 'b': 4}, {'a': 3, 'b': 4}, {'a': 1, 'b': 5}, {'a': 2, 'b': 5},
                    {'a': 3, 'b': 5}]
    shouldBeListOfStrings = [str(curHash) for curHash in shouldBeList]
    returnedListOfStrings = [str(curHash) for curHash in returned]
    assert set(shouldBeListOfStrings) == set(returnedListOfStrings)


def test_ArgumentIterator():
    """Assert that argument iterator can handle empty arguments and
    iterates as otherwise expected """
    #assert works with empty dict
    returned = ArgumentIterator({})
    assert {} == next(returned)
    #should be out of iterations after popping the empty list
    try:
        next(returned)
        #if next works a second time, then ArgumentIterator has too many iterations
        assert False
    except StopIteration:
        pass

    argumentDict = {'a': CV([1, 2, 3]), 'b': CV([4, 5])}
    argIter = ArgumentIterator(argumentDict)

    iterationCount = 0
    for curArgumentCombo in argIter:
        iterationCount += 1
        assert set(argumentDict.keys()) == set(curArgumentCombo.keys())
        assert len(list(argumentDict.keys())) == len(list(curArgumentCombo.keys()))

    assert iterationCount == len(argIter.permutationsList)


def test_ArgumentIterator_stringsAndTuples():
    arguments = {'a': 'hello', 'b': CV([1, 2, 5])}

    returned = ArgumentIterator(arguments)

    for curr in returned:
        assert curr['a'] == 'hello'
        assert curr['b'] in (1, 2, 5)

        assert len(list(curr.keys())) == 2


def test_ArgumentIterator_separateResults():
    arguments = {'a': 'hello', 'b': CV([1, 2, 5])}

    argIter = ArgumentIterator(arguments)

    rets = []
    for curr in argIter:
        rets.append(curr)

    retsCopy = copy.deepcopy(rets)

    for i in range(len(rets)):
        assert rets[i] == retsCopy[i]

    for i in range(len(rets)):
        rets[i]['a'] = i
        for j in range(i + 1, len(rets)):
            assert rets[i]['a'] != rets[j]['a']
            assert rets[j] == retsCopy[j]
