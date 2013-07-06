"""
Unit tests for shogun_interface.py

"""

import numpy
import scipy.sparse
from numpy.random import rand, randint
from nose.tools import *

from UML.exceptions import ArgumentException

from UML.interfaces import shogun
from UML.interfaces.shogun_interface import listAlgorithms
from UML.interfaces.shogun_interface import setShogunLocation
from UML.interfaces.shogun_interface import getShogunLocation
from UML.data.dense_matrix_data import DenseMatrixData as DMData
from UML.data.coo_sparse_data import CooSparseData


def testShogunLocation():
	""" Test setShogunLocation() """
	path = '/test/path/shogun'
	setShogunLocation(path)

	assert getShogunLocation() == path

@raises(ArgumentException)
def testShogun_shapemismatchException():
	""" Test shogun() raises exception when the shape of the train and test data don't match """
	variables = ["Y","x1","x2"]
	data = [[-1,1,0], [-1,0,1], [1,3,2]]
	trainingObj = DMData(data,variables)

	data2 = [[3]]
	testObj = DMData(data2)

	args = {}
	ret = shogun("LibLinear", trainingObj, testObj, output=None, dependentVar="Y", arguments=args)


@raises(ArgumentException)
def testShogun_singleClassException():
	""" Test shogun() raises exception when the training data only has a single label """
	variables = ["Y","x1","x2"]
	data = [[-1,1,0], [-1,0,1], [1,3,2]]
	trainingObj = DMData(data,variables)

	data2 = [[3]]
	testObj = DMData(data2)

	args = {}
	ret = shogun("LibLinear", trainingObj, testObj, output=None, dependentVar="Y", arguments=args)



def testShogunHandmadeBinaryClassification():
	""" Test shogun() by calling a binary linear classification algorithm"""
	variables = ["Y","x1","x2"]
	data = [[0,1,0], [-0,0,1], [1,3,2]]
	trainingObj = DMData(data,variables)

	data2 = [[3,3], [-1,0]]
	testObj = DMData(data2)

	args = {}
	ret = shogun("LibLinear", trainingObj, testObj, output=None, dependentVar="Y", arguments=args)

	assert ret is not None

	# shogun binary classifiers seem to return confidence values, not class ID
	assert ret.data[0,0] > 0

def testShogunHandmadeBinaryClassificationWithKernel():
	""" Test shogun() by calling a binary linear classification algorithm with a kernel """
	variables = ["Y","x1","x2"]
	data = [[5,-11,-5], [1,0,1], [1,3,2]]
	trainingObj = DMData(data,variables)

	data2 = [[5,3], [-1,0]]
	testObj = DMData(data2)

	args = {'kernel':'GaussianKernel', 'width':2, 'size':10}
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
	data = [[0,0,0], [0,0,1], [1,-118,1], [1,-117,1], [2,1,191], [2,1,118], [3,-1000,-500]]
	trainingObj = DMData(data,variables)

	data2 = [[0,0], [-101,1], [1,101], [1,1]]
	testObj = DMData(data2)

	args = {'C':.5, 'kernel':'LinearKernel'}
#	args = {'C':1}
	ret = shogun("MulticlassLibSVM", trainingObj, testObj, output=None, dependentVar="Y", arguments=args)

	assert ret is not None

	assert ret.data[0,0] == 0
	assert ret.data[1,0] == 1
	assert ret.data[2,0] == 2


def testShogunSparseRegression():
	""" Test shogun() sparse data instantiation by calling a sparse regression algorithm with a large, but highly sparse, matrix """

	x = 100
	c = 10
	points = randint(0,x,c)
	cols = randint(0,x,c)
	data = rand(c)
	A = scipy.sparse.coo_matrix( (data, (points,cols)), shape=(x,x))
	obj = CooSparseData(A)

	labelsData = numpy.random.rand(x)
	labels = DMData(labelsData)

	ret = shogun('MulticlassOCAS', trainData=obj, testData=obj, dependentVar=labels)

	assert ret is not None


def testShogunRossData():
	""" Test shogun() by calling classifers using the problematic data from Ross """
	
	p0 = [1,  0,    0,    0,    0.21,  0.12]
	p1 = [2,  0,    0.56, 0.77, 0,     0]
	p2 = [1,  0.24, 0,    0,    0.12,  0]
	p3 = [1,  0,    0,    0,    0,     0.33]
	p4 = [2,  0.55, 0,    0.67, 0.98,  0]
	p5 = [1,  0,    0,    0,    0.21,  0.12]
	p6 = [2,  0,    0.56, 0.77, 0,     0]
	p7 = [1,  0.24, 0,    0,    0.12,  0]

	data = [p0,p1,p2,p3,p4,p5,p6,p7]

	trainingObj = DMData(data)

	data2 = [[0, 0, 0, 0, 0.33], [0.55, 0, 0.67, 0.98, 0]]
	testObj = DMData(data2)

	args = {'C':1.0}
	argsk = {'C':1.0, 'kernel':"LinearKernel"}

	ret = shogun("MulticlassLibSVM", trainingObj, testObj, output=None, dependentVar=0, arguments=argsk)
	assert ret is not None

	ret = shogun("MulticlassLibLinear", trainingObj, testObj, output=None, dependentVar=0, arguments=args)
	assert ret is not None

	ret = shogun("LaRank", trainingObj, testObj, output=None, dependentVar=0, arguments=argsk)
	assert ret is not None

	ret = shogun("MulticlassOCAS", trainingObj, testObj, output=None, dependentVar=0, arguments=args)
	assert ret is not None


def testShogunEmbeddedRossData():
	""" Test shogun() by MulticlassOCAS with the ross data embedded in random data """
	
	p0 = [3,  0,    0,    0,    0.21,  0.12]
	p1 = [2,  0,    0.56, 0.77, 0,     0]
	p2 = [3,  0.24, 0,    0,    0.12,  0]
	p3 = [3,  0,    0,    0,    0,     0.33]
	p4 = [2,  0.55, 0,    0.67, 0.98,  0]
	p5 = [3,  0,    0,    0,    0.21,  0.12]
	p6 = [2,  0,    0.56, 0.77, 0,     0]
	p7 = [3,  0.24, 0,    0,    0.12,  0]

	data = [p0,p1,p2,p3,p4,p5,p6,p7]

	numpyData = numpy.zeros((50,10))

	for i in xrange(50):
		for j in xrange(10):
			if i < 8 and j < 6:
				numpyData[i,j] = data[i][j]
			else:
				if j == 0:
					numpyData[i,j] = numpy.random.randint(2,3)
				else:
					numpyData[i,j] = numpy.random.rand()

	trainingObj = DMData(numpyData)

	data2 = [[0, 0, 0, 0, 0.33,0, 0, 0, 0.33], [0.55, 0, 0.67, 0.98,0.55, 0, 0.67, 0.98, 0]]
	testObj = DMData(data2)

	args = {'C':1.0}

	ret = shogun("MulticlassOCAS", trainingObj, testObj, output=None, dependentVar=0, arguments=args)
	assert ret is not None

	for value in ret.data:
		assert value == 2 or value == 3


def testShogunScoreModeMulti():
	""" Test shogun() returns the right dimensions when given different scoreMode flags, multi case"""
	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2]]
	trainingObj = DMData(data,variables)

	data2 = [[2,3],[-200,0]]
	testObj = DMData(data2)

	# default scoreMode is 'label'
	ret = shogun("MulticlassOCAS", trainingObj, testObj, dependentVar="Y", arguments={})
	assert ret.points() == 2
	assert ret.features() == 1

	ret = shogun("MulticlassOCAS", trainingObj, testObj, dependentVar="Y", arguments={}, scoreMode='bestScore')
	assert ret.points() == 2
	assert ret.features() == 2

	ret = shogun("MulticlassOCAS", trainingObj, testObj, dependentVar="Y", arguments={}, scoreMode='allScores')
	assert ret.points() == 2
	assert ret.features() == 3


def testShogunScoreModeBinary():
	""" Test shogun() returns the right dimensions when given different scoreMode flags, binary case"""
	variables = ["Y","x1","x2"]
	data = [[-1,1,1], [-1,0,1], [1,30,2], [1,30,3]]
	trainingObj = DMData(data,variables)

	data2 = [[2,1],[25,0]]
	testObj = DMData(data2)

	# default scoreMode is 'label'
	ret = shogun("SVMOcas", trainingObj, testObj, dependentVar="Y", arguments={})
	assert ret.points() == 2
	assert ret.features() == 1

	ret = shogun("SVMOcas", trainingObj, testObj, dependentVar="Y", arguments={}, scoreMode='bestScore')
	assert ret.points() == 2
	assert ret.features() == 2

	ret = shogun("SVMOcas", trainingObj, testObj, dependentVar="Y", arguments={}, scoreMode='allScores')
	assert ret.points() == 2
	assert ret.features() == 2


def testShogunMultiClassStrategyMultiDataBinaryAlg():
	""" Test shogun() will correctly apply the provided strategies when given multiclass data and a binary algorithm"""
	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2]]
	trainingObj = DMData(data,variables)

	data2 = [[2,3],[-200,0]]
	testObj = DMData(data2)

#	import pdb
#	pdb.set_trace()

	ret = shogun("SVMOcas", trainingObj, testObj, dependentVar="Y", arguments={}, multiClassStrategy="OneVsOne")
	




def testShogunListAlgorithms():
	""" Test shogun's listAlgorithms() by checking the output for those algorithms we unit test """

	ret = listAlgorithms()

	assert 'LibSVM' in ret
	assert 'LibLinear' in ret
	assert 'MulticlassLibSVM' in ret
	assert 'SubGradientSVM' in ret
	assert 'MulticlassOCAS' in ret
