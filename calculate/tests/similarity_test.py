import numpy
import math

from UML import createData
from UML.calculate import cosineSimilarity

####################
# cosineSimilarity #
####################

def test_cosineSimilarity():
	orig = numpy.array([[1],[0]])
	orth = numpy.array([[0],[1]])
	neg = numpy.array([[-1],[0]])

	origMatrix = createData('Matrix', data=orig)
	orthMatrix = createData('Matrix', data=orth)
	negMatrix = createData('Matrix', data=neg)

	result0 = cosineSimilarity(origMatrix, origMatrix)
	result1 = cosineSimilarity(origMatrix, orthMatrix)
	result2 = cosineSimilarity(origMatrix, negMatrix)

	assert result0 == 1
	assert result1 == 0
	assert result2 == -1

def test_cosineSimilarityZeros():
	zeros = [[0],[0]]

	zerosMatrix = createData('Matrix', data=zeros)

	result0 = cosineSimilarity(zerosMatrix, zerosMatrix)

	assert math.isnan(result0)
