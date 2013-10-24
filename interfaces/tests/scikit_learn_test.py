"""
Unit tests for scikit_learn_interface.py

"""

import numpy.testing
import scipy.sparse
from numpy.random import rand, randint

from UML.interfaces.tests.test_helpers import checkLabelOrderingAndScoreAssociations
from UML.interfaces.scikit_learn_interface import *
from UML.data import Matrix
from UML.data import Sparse



def testSciKitLearnLocation():
	""" Test setSciKitLearnLocation() """
	path = '/test/path/skl'
	setSciKitLearnLocation(path)

	assert getSciKitLearnLocation() == path


def testSciKitLearnHandmadeRegression():
	""" Test sciKitLearn() by calling a regression algorithm with known output """
	variables = ["Y","x1","x2"]
	data = [[2,1,1], [3,1,2], [4,2,2],]
	trainingObj = Matrix(data,variables)

	data2 = [[0,1]]
	testObj = Matrix(data2)

	ret = sciKitLearn("LinearRegression", trainingObj, trainY="Y", testX=testObj, output=None, arguments={})

	assert ret is not None

	expected = [[1.]]
	expectedObj = Matrix(expected)

	numpy.testing.assert_approx_equal(ret.data[0,0],1.)

def testSciKitLearnSparseRegression():
	""" Test sciKitLearn() by calling a sparse regression algorithm with an extremely large, but highly sparse, matrix """

	x = 100000
	c = 10
	points = randint(0,x,c)
	points2 = randint(0,x,c)
	cols = randint(0,x,c)
	cols2 = randint(0,x,c)
	data = rand(c)
	A = scipy.sparse.coo_matrix( (data, (points,cols)), shape=(x,x))
	obj = Sparse(A)
	testObj = obj.copy()
	testObj.extractFeatures(cols[0])

	ret = sciKitLearn('SGDRegressor', trainX=obj, trainY=cols[0], testX=testObj)

	assert ret is not None

def testSciKitLearnHandmadeClustering():
	""" Test sciKitLearn() by calling a clustering algorithm with known output """
	variables = ["x1","x2"]
	data = [[1,0], [3,3], [5,0],]
	trainingObj = Matrix(data,variables)

	data2 = [[1,0],[1,1],[5,1], [3,4]]
	testObj = Matrix(data2)

	ret = sciKitLearn("KMeans", trainingObj, testX=testObj, output=None, arguments={'n_clusters':3})

	# clustering returns a row vector of indices, referring to the cluster centers,
	# we don't care about the exact numbers, this verifies that the appropriate
	# ones are assigned to the same clusters
	assert ret.data[0,0] == ret.data[1,0]
	assert ret.data[0,0] != ret.data[2,0]
	assert ret.data[0,0] != ret.data[3,0]
	assert ret.data[2,0] != ret.data[3,0]


def testSciKitLearnHandmadeSparseClustering():
	""" Test sciKitLearn() by calling a sparse clustering algorithm with known output """
	trainData = scipy.sparse.lil_matrix((3, 3))
	trainData[0, :] = [2,3,1]
	trainData[1, :] = [2,2,1]
	trainData[2, :] = [0,0,0]
	trainData = Sparse(data=trainData)

	testData = scipy.sparse.lil_matrix((3,2))
	testData[0, :] = [3,3]
	testData[1, :] = [3,2]
	testData[2, :] = [-1,0]
	testData = Sparse(data=testData)

	ret = sciKitLearn('MiniBatchKMeans', trainData, trainY=2, testX=testData, arguments={'n_clusters':2})
	
	assert ret[0,0] == ret[1,0]
	assert ret[0,0] != ret[2,0]


def testSciKitLearnScoreMode():
	""" Test sciKitLearn() scoreMode flags"""
	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2]]
	trainingObj = Matrix(data,variables)

	data2 = [[2,3],[-200,0]]
	testObj = Matrix(data2)

	# default scoreMode is 'label'
	ret = sciKitLearn("SVC", trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	bestScores = sciKitLearn("SVC", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert bestScores.pointCount == 2
	assert bestScores.featureCount == 2

	allScores = sciKitLearn("SVC", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
	assert allScores.pointCount == 2
	assert allScores.featureCount == 3

	checkLabelOrderingAndScoreAssociations([0,1,2], bestScores, allScores)


def testSciKitLearnScoreModeBinary():
	""" Test sciKitLearn() scoreMode flags, binary case"""
	variables = ["Y","x1","x2"]
	data = [[1,30,2],[2,1,1], [2,0,1],[2,-1,-1],  [1,30,3], [1,34,4]]
	trainingObj = Matrix(data,variables)

	data2 = [[2,1],[25,0]]
	testObj = Matrix(data2)

	# default scoreMode is 'label'
	ret = sciKitLearn("SVC", trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	bestScores = sciKitLearn("SVC", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert bestScores.pointCount == 2
	assert bestScores.featureCount == 2

	allScores = sciKitLearn("SVC", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
	assert allScores.pointCount == 2
	assert allScores.featureCount == 2

	checkLabelOrderingAndScoreAssociations([1,2], bestScores, allScores)


def testSciKitLearnListAlgorithms():
	""" Test scikit learn's listSciKitLearnAlgorithms() by checking the output for those algorithms we unit test """

	ret = listSciKitLearnAlgorithms()

	assert 'KMeans' in ret
	assert 'LinearRegression' in ret

	toExclude = ['BaseDiscreteNB', 'GaussianNB', 'libsvm']

	for name in ret:
		if name not in toExclude:
			params = getParameters(name)
			assert params is not None
			defaults = getDefaultValues(name)
			for key in defaults.keys():
				assert key in params



