"""
Unit tests for scikit_learn_interface.py

"""

import numpy.testing
import scipy.sparse
from numpy.random import rand, randint

from ..scikit_learn_interface import *
from ...processing.dense_matrix_data import DenseMatrixData as DMData
from ...processing.coo_sparse_data import CooSparseData
from ...processing.csc_sparse_data import CscSparseData



def testSciKitLearnLocation():
	""" Test setSciKitLearnLocation() """
	path = '/test/path/skl'
	setSciKitLearnLocation(path)

	assert getSciKitLearnLocation() == path


def testSciKitLearnHandmadeRegression():
	""" Test sciKitLearn() by calling a regression algorithm with known output """
	variables = ["Y","x1","x2"]
	data = [[2,1,1], [3,1,2], [4,2,2],]
	trainingObj = DMData(data,variables)

	data2 = [[0,1]]
	testObj = DMData(data2)

	ret = sciKitLearn("LinearRegression", trainingObj, testObj, output=None, dependentVar="Y", arguments={})

	assert ret is not None

	expected = [[1.]]
	expectedObj = DMData(expected)

	numpy.testing.assert_approx_equal(ret.data[0,0],1.)

def testSciKitLearnSparseRegression():
	""" Test sciKitLearn() by calling a sparse regression algorithm with an extremely large, but highly sparse, matrix """

	x = 100000
	c = 10
	rows = randint(0,x,c)
	rows2 = randint(0,x,c)
	cols = randint(0,x,c)
	cols2 = randint(0,x,c)
	data = rand(c)
	A = scipy.sparse.coo_matrix( (data, (rows,cols)), shape=(x,x))
	obj = CooSparseData(A)

	ret = sciKitLearn('SGDRegressor', trainData=obj, testData=obj, dependentVar=cols[0])

	assert ret is not None

def testSciKitLearnHandmadeClustering():
	""" Test sciKitLearn() by calling a clustering algorithm with known output """
	variables = ["x1","x2"]
	data = [[1,0], [3,3], [5,0],]
	trainingObj = DMData(data,variables)

	data2 = [[1,0],[1,1],[5,1], [3,4]]
	testObj = DMData(data2)

	ret = sciKitLearn("KMeans", trainingObj, testObj, output=None, arguments={'n_clusters':3})

	# clustering returns a row vector of indices, referring to the cluster centers,
	# we don't care about the exact numbers, this verifies that the appropriate
	# ones are assigned to the same clusters
	assert ret.data[0,0] == ret.data[0,1]
	assert ret.data[0,0] != ret.data[0,2]
	assert ret.data[0,0] != ret.data[0,3]
	assert ret.data[0,2] != ret.data[0,3]


def testSciKitLearnHandmadeSparseClustering():
	""" Test sciKitLearn() by calling a sparse clustering algorithm with known output """
	trainData = scipy.sparse.lil_matrix((3, 3))
	trainData[0, :] = [2,3,1]
	trainData[1, :] = [2,2,1]
	trainData[2, :] = [0,0,0]
	trainData = CscSparseData(data=trainData)

	testData = scipy.sparse.lil_matrix((3,2))
	testData[0, :] = [3,3]
	testData[1, :] = [3,2]
	testData[2, :] = [-1,0]
	testData = CscSparseData(data=testData)

	ret = sciKitLearn('MiniBatchKMeans', trainData, testData, dependentVar=2, arguments={'n_clusters':2})
	
	assert ret.data[0,0] == ret.data[0,1]
	assert ret.data[0,0] != ret.data[0,2]


def testSciKitLearnListAlgorithms():
	""" Test scikit learn's listAlgorithms() by checking the output for those algorithms we unit test """

	ret = listAlgorithms()

	assert 'KMeans' in ret
	assert 'LinearRegression' in ret



