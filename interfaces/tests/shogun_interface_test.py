"""
Unit tests for shogun_interface.py

"""

try:
	import clang
	clangAvailable = True
except ImportError:
	clangAvailable = False

import numpy
import scipy.sparse
from UML.randomness import numpyRandom
from nose.tools import *
from nose.plugins.attrib import attr

import UML

from UML.exceptions import ArgumentException


@raises(ArgumentException)
def testShogun_shapemismatchException():
	""" Test shogun raises exception when the shape of the train and test data don't match """
	variables = ["Y","x1","x2"]
	data = [[-1,1,0], [-1,0,1], [1,3,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[3]]
	testObj = UML.createData('Matrix', data2)

	args = {}
	ret = UML.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)


@raises(ArgumentException)
def testShogun_singleClassException():
	""" Test shogun raises exception when the training data only has a single label """
	variables = ["Y","x1","x2"]
	data = [[-1,1,0], [-1,0,1], [-1,0,0]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[3,3]]
	testObj = UML.createData('Matrix', data2)

	args = {}
	ret = UML.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

@raises(ArgumentException)
def testShogun_multiClassDataToBinaryAlg():
	""" Test shogun() raises ArgumentException when passing multiclass data to a binary classifier """
	variables = ["Y","x1","x2"]
	data = [[5,-11,-5], [1,0,1], [2,3,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[5,3], [-1,0]]
	testObj = UML.createData('Matrix', data2)

	args = {'kernel':'GaussianKernel', 'width':2, 'size':10}
	# TODO -  is this failing because of kernel issues, or the thing we want to test?
	ret = UML.trainAndApply("shogun.LibSVM", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)


def testShogunHandmadeBinaryClassification():
	""" Test shogun by calling a binary linear classifier """
	variables = ["Y","x1","x2"]
	data = [[0,1,0], [-0,0,1], [1,3,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[3,3], [-1,0]]
	testObj = UML.createData('Matrix', data2)

	args = {}
	ret = UML.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

	assert ret is not None

	# shogun binary classifiers seem to return confidence values, not class ID
	assert ret.data[0,0] > 0

def testShogunHandmadeBinaryClassificationWithKernel():
	""" Test shogun by calling a binary linear classifier with a kernel """
	if not clangAvailable:
		return

	variables = ["Y","x1","x2"]
	data = [[5,-11,-5], [1,0,1], [1,3,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[5,3], [-1,0]]
	testObj = UML.createData('Matrix', data2)

	args = {'st':1, 'kernel':'GaussianKernel', 'w':2, 'size':10}
	ret = UML.trainAndApply("shogun.LibSVM", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

	assert ret is not None

	# shogun binary classifiers seem to return confidence values, not class ID
	assert ret.data[0,0] > 0

def testShogunKMeans():
	""" Test shogun by calling the Kmeans classifier, a distance based machine """
	variables = ["Y","x1","x2"]
	data = [[0,0,0], [0,0,1], [1,8,1], [1,7,1], [2,1,9], [2,1,8]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[0,-10], [10,1], [1,10]]
	testObj = UML.createData('Matrix', data2)

	args = {'distance':'ManhattanMetric'}

	ret = UML.trainAndApply("shogun.KNN", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

	assert ret is not None

	assert ret.data[0,0] == 0
	assert ret.data[1,0] == 1
	assert ret.data[2,0] == 2


def testShogunMulticlassSVM():
	""" Test shogun by calling a multilass classifier with a kernel """
	if not clangAvailable:
		return

	variables = ["Y","x1","x2"]
	data = [[0,0,0], [0,0,1], [1,-118,1], [1,-117,1], [2,1,191], [2,1,118], [3,-1000,-500]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[0,0], [-101,1], [1,101], [1,1]]
	testObj = UML.createData('Matrix', data2)

	args = {'C':.5, 'k':'LinearKernel'}

#	args = {'C':1}
#	args = {}
	ret = UML.trainAndApply("shogun.GMNPSVM", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

	assert ret is not None

	assert ret.data[0,0] == 0
	assert ret.data[1,0] == 1
	assert ret.data[2,0] == 2


def testShogunSparseRegression():
	""" Test shogun sparse data instantiation by calling on a sparse regression learner with a large, but highly sparse, matrix """

	x = 100
	c = 10
	points = numpyRandom.randint(0,x,c)
	cols = numpyRandom.randint(0,x,c)
	data = numpyRandom.rand(c)
	A = scipy.sparse.coo_matrix( (data, (points,cols)), shape=(x,x))
	obj = UML.createData('Sparse', A)

	labelsData = numpyRandom.rand(x)
	labels = UML.createData('Matrix', labelsData.reshape((x,1)))

	ret = UML.trainAndApply('shogun.MulticlassOCAS', trainX=obj, trainY=labels, testX=obj, max_train_time=10)

	assert ret is not None


def testShogunRossData():
	""" Test shogun by calling classifers using the problematic data from Ross """
	if not clangAvailable:
		return

	p0 = [1,  0,    0,    0,    0.21,  0.12]
	p1 = [2,  0,    0.56, 0.77, 0,     0]
	p2 = [1,  0.24, 0,    0,    0.12,  0]
	p3 = [1,  0,    0,    0,    0,     0.33]
	p4 = [2,  0.55, 0,    0.67, 0.98,  0]
	p5 = [1,  0,    0,    0,    0.21,  0.12]
	p6 = [2,  0,    0.56, 0.77, 0,     0]
	p7 = [1,  0.24, 0,    0,    0.12,  0]

	data = [p0,p1,p2,p3,p4,p5,p6,p7]

	trainingObj = UML.createData('Matrix', data)

	data2 = [[0, 0, 0, 0, 0.33], [0.55, 0, 0.67, 0.98, 0]]
	testObj = UML.createData('Matrix', data2)

	args = {'C':1.0}
	argsk = {'C':1.0, 'k':"LinearKernel"}

	ret = UML.trainAndApply("shogun.MulticlassLibSVM", trainingObj, trainY=0, testX=testObj, output=None, arguments=argsk)
	assert ret is not None

	ret = UML.trainAndApply("shogun.MulticlassLibLinear", trainingObj, trainY=0, testX=testObj, output=None, arguments=args)
	assert ret is not None

	ret = UML.trainAndApply("shogun.LaRank", trainingObj, trainY=0, testX=testObj, output=None, arguments=argsk)
	assert ret is not None

	ret = UML.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY=0, testX=testObj, output=None, arguments=args)
	assert ret is not None

@attr('slow')
def testShogunEmbeddedRossData():
	""" Test shogun by MulticlassOCAS with the ross data embedded in random data """
	
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
					numpyData[i,j] = numpyRandom.randint(2,3)
				else:
					numpyData[i,j] = numpyRandom.rand()

	trainingObj = UML.createData('Matrix', numpyData)

	data2 = [[0, 0, 0, 0, 0.33,0, 0, 0, 0.33], [0.55, 0, 0.67, 0.98,0.55, 0, 0.67, 0.98, 0]]
	testObj = UML.createData('Matrix', data2)

	args = {'C':1.0}

	ret = UML.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY=0, testX=testObj, output=None, arguments=args)
	assert ret is not None

	for value in ret.data:
		assert value == 2 or value == 3


def testShogunScoreModeMulti():
	""" Test shogun returns the right dimensions when given different scoreMode flags, multi case"""
	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[2,3],[-200,0]]
	testObj = UML.createData('Matrix', data2)

	# default scoreMode is 'label'
	ret = UML.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	ret = UML.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert ret.pointCount == 2
	assert ret.featureCount == 2

	ret = UML.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
	assert ret.pointCount == 2
	assert ret.featureCount == 3


def testShogunScoreModeBinary():
	""" Test shogun returns the right dimensions when given different scoreMode flags, binary case"""
	variables = ["Y","x1","x2"]
	data = [[-1,1,1], [-1,0,1], [1,30,2], [1,30,3]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[2,1],[25,0]]
	testObj = UML.createData('Matrix', data2)

	# default scoreMode is 'label'
	ret = UML.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	ret = UML.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert ret.pointCount == 2
	assert ret.featureCount == 2

	ret = UML.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
	assert ret.pointCount == 2
	assert ret.featureCount == 2

def onlineLearneres():
#def testOnlineLearners():
	""" Test shogun can call online learners """
	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [0,3,2], [1,-300,-25]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[2,3],[-200,0]]
	testObj = UML.createData('Matrix', data2)

	ret = UML.trainAndApply("shogun.OnlineLibLinear", trainingObj, trainY="Y", testX=testObj, arguments={})
	ret = UML.trainAndApply("shogun.OnlineSVMSGD", trainingObj, trainY="Y", testX=testObj, arguments={})
		

# TODO def testShogunMultiClassStrategyMultiDataBinaryAlg():
def notRunnable():
	""" Test shogun will correctly apply the provided strategies when given multiclass data and a binary learner """
	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[2,3],[-200,0]]
	testObj = UML.createData('Matrix', data2)

	ret = UML.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={}, multiClassStrategy="OneVsOne")
	



@attr('slow')
def testShogunListLearners():
	""" Test shogun's listShogunLearners() by checking the output for those learners we unit test """

	ret = UML.listLearners('shogun')

	assert 'LibSVM' in ret
	assert 'LibLinear' in ret
	assert 'MulticlassLibSVM' in ret
	assert 'MulticlassOCAS' in ret

	for name in ret:
		params = UML.learnerParameters('shogun.' + name)
		assert params is not None
		defaults = UML.learnerDefaultValues('shogun.' + name)
		for pSet in params:
			for dSet in defaults:
				for key in dSet.keys():
					assert key in pSet

