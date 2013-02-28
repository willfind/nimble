"""
Unit tests for shogun_interface.py

"""

import numpy
import scipy.sparse
from numpy.random import rand, randint

from ..shogun_interface import shogun
from ..shogun_interface import listAlgorithms
from ..shogun_interface import setShogunLocation
from ..shogun_interface import getShogunLocation
from ...processing.dense_matrix_data import DenseMatrixData as DMData
from ...processing.coo_sparse_data import CooSparseData
from ...processing.csc_sparse_data import CscSparseData



def testShogunLocation():
	""" Test setShogunLocation() """
	path = '/test/path/shogun'
	setShogunLocation(path)

	assert getShogunLocation() == path


def testShogunHandmadeBinaryClassification():
	""" Test shogun() by calling a binary linear classification algorithm"""
	variables = ["Y","x1","x2"]
	data = [[-1,1,0], [-1,0,1], [1,3,2]]
	trainingObj = DMData(data,variables)

	data2 = [[3,3]]
	testObj = DMData(data2)

	args = {}
	ret = shogun("LibLinear", trainingObj, testObj, output=None, dependentVar="Y", arguments=args)

	assert ret is not None

	# shogun binary classifiers seem to return confidence values, not class ID
	assert ret.data[0,0] > 0

def testShogunHandmadeBinaryClassificationWithKernel():
	""" Test shogun() by calling a binary linear classification algorithm with a kernel """
	variables = ["Y","x1","x2"]
	data = [[-1,1,0], [-1,0,1], [1,3,2]]
	trainingObj = DMData(data,variables)

	data2 = [[3,3]]
	testObj = DMData(data2)

	args = {'kernel':'GaussianKernel', 'width':2}
	ret = shogun("LibSVM", trainingObj, testObj, output=None, dependentVar="Y", arguments=args)

	assert ret is not None

	# shogun binary classifiers seem to return confidence values, not class ID
	assert ret.data[0,0] > 0

def testShogunKMeans():
	""" Test shogun() by calling the Kmeans classifier, a distance based machine """
	variables = ["Y","x1","x2"]
	data = [[0,0,0], [0,0,1], [1,8,1], [1,7,1], [2,1,9], [2,1,8]]
	trainingObj = DMData(data,variables)

	data2 = [[0,-10], [10,1], [1,10]]
	testObj = DMData(data2)

	args = {'distance':'ManhattanMetric'}
	ret = shogun("KNN", trainingObj, testObj, output=None, dependentVar="Y", arguments=args)

	assert ret is not None

	assert ret.data[0,0] == 0
	assert ret.data[1,0] == 1
	assert ret.data[2,0] == 2


def testShogunMulticlassSVM():
	""" Test shogun() by calling a multilass classifier with a kernel """
	variables = ["Y","x1","x2"]
	data = [[0,0,0], [0,0,1], [1,8,1], [1,7,1], [2,1,9], [2,1,8]]
	trainingObj = DMData(data,variables)

	data2 = [[0,0], [10,1], [1,10]]
	testObj = DMData(data2)

	args = {'C':.5, 'kernel':'PolyKernel', 'd':2, 'inhom':False}
	ret = shogun("LibSVMMultiClass", trainingObj, testObj, output=None, dependentVar="Y", arguments=args)

	assert ret is not None

	# shogun binary classifiers seem to return confidence values, not class ID
	assert ret.data[0,0] == 0
	assert ret.data[1,0] == 1
	assert ret.data[2,0] == 2


def testShogunSparseRegression():
	""" Test shogun() by calling a sparse regression algorithm with an extremely large, but highly sparse, matrix """

	x = 100000
	c = 10
	points = randint(0,x,c)
	cols = randint(0,x,c)
	data = rand(c)
	A = scipy.sparse.coo_matrix( (data, (points,cols)), shape=(x,x))
	obj = CooSparseData(A)

	labelsData = numpy.random.rand(x)
	labels = DMData(labelsData)

	ret = shogun('SubGradientSVM', trainData=obj, testData=obj, dependentVar=labels)

	assert ret is not None


def testShogunListAlgorithms():
	""" Test shogun's listAlgorithms() by checking the output for those algorithms we unit test """

	ret = listAlgorithms()

	assert 'LibSVM' in ret
	assert 'LibLinear' in ret
	assert 'LibSVMMultiClass' in ret
	assert 'SubGradientSVM' in ret
