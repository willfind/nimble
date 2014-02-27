"""
Unit tests for scikit_learn_interface.py

"""

import numpy.testing
import scipy.sparse
from numpy.random import rand, randint

import UML

from UML.interfaces.tests.test_helpers import checkLabelOrderingAndScoreAssociations
from UML.data import Matrix
from UML.data import Sparse

packageName = 'sciKitLearn'

def toCall(learner):
	return packageName + '.' + learner 


def testSciKitLearnHandmadeRegression():
	""" Test sciKitLearn() by calling on a regression learner with known output """
	variables = ["Y","x1","x2"]
	data = [[2,1,1], [3,1,2], [4,2,2],]
	trainingObj = Matrix(data,variables)

	data2 = [[0,1]]
	testObj = Matrix(data2)

	ret = UML.run(toCall("LinearRegression"), trainingObj, trainY="Y", testX=testObj, output=None, arguments={})

	assert ret is not None

	expected = [[1.]]
	expectedObj = Matrix(expected)

	numpy.testing.assert_approx_equal(ret[0,0],1.)

def testSciKitLearnSparseRegression():
	""" Test sciKitLearn() by calling on a sparse regression learner with an extremely large, but highly sparse, matrix """

	x = 1000
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

	ret = UML.run(toCall('SGDRegressor'), trainX=obj, trainY=cols[0], testX=testObj)

	assert ret is not None

def testSciKitLearnHandmadeClustering():
	""" Test sciKitLearn() by calling a clustering classifier with known output """
	variables = ["x1","x2"]
	data = [[1,0], [3,3], [5,0],]
	trainingObj = Matrix(data,variables)

	data2 = [[1,0],[1,1],[5,1], [3,4]]
	testObj = Matrix(data2)

	ret = UML.run(toCall("KMeans"), trainingObj, testX=testObj, output=None, arguments={'n_clusters':3})

	# clustering returns a row vector of indices, referring to the cluster centers,
	# we don't care about the exact numbers, this verifies that the appropriate
	# ones are assigned to the same clusters
	assert ret[0,0] == ret[1,0]
	assert ret[0,0] != ret[2,0]
	assert ret[0,0] != ret[3,0]
	assert ret[2,0] != ret[3,0]


def testSciKitLearnHandmadeSparseClustering():
	""" Test sciKitLearn() by calling on a sparse clustering learner with known output """
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

	ret = UML.run(toCall('MiniBatchKMeans'), trainData, trainY=2, testX=testData, arguments={'n_clusters':2})
	
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
	ret = UML.run(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	bestScores = UML.run(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert bestScores.pointCount == 2
	assert bestScores.featureCount == 2

	allScores = UML.run(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
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
	ret = UML.run(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	bestScores = UML.run(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert bestScores.pointCount == 2
	assert bestScores.featureCount == 2

	allScores = UML.run(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
	assert allScores.pointCount == 2
	assert allScores.featureCount == 2

	checkLabelOrderingAndScoreAssociations([1,2], bestScores, allScores)


def testSciKitLearnUnsupervisedProblemLearners():
	""" Test sciKitLearn() by calling some unsupervised learners problematic in previous implementations """
	variables = ["x1","x2"]
	data = [[1,0], [3,3], [50,0]]
	trainingObj = Matrix(data,variables)

	data2 = [[1,0],[1,1],[5,1], [34,4]]
	testObj = Matrix(data2)

	UML.run(toCall("GMM"), trainingObj, testX=testObj, arguments={'n_components':3})
	UML.run(toCall("DPGMM"), trainingObj, testX=testObj)
	UML.run(toCall("VBGMM"), trainingObj, testX=testObj)


def testSciKitLearnObsAsArgumentName():
	""" Test scikitLearn() by calling learners with 'obs' instead of 'X' as a fit/predict argument """
	variables = ["x1","x2", "x3"]
	data = [[1,3,3], [6,7,6], [50,1,3]]
	trainingObj = Matrix(data,variables)

	data2 = [[2,1],[1,2],[5,1], [34,4]]
	testObj = Matrix(data2)

	ret = UML.run(toCall("GMMHMM"), trainingObj, testX=testObj, arguments={'n_components':3})
	ret = UML.run(toCall("GaussianHMM"), trainingObj, testX=testObj)
	ret = UML.run(toCall("MultinomialHMM"), trainingObj, testX=testObj)

def testSciKitLearnArgspecFailures():
	""" Test scikitLearn() on those learners that cannot be passed to inspect.getargspec """
	variables = ["x1","x2"]
	data = [[1,0], [3,3], [50,0]]
	trainingObj = Matrix(data,variables)

	dataY = [[0],[1],[2]]
	trainingYObj = Matrix(dataY)

	data2 = [[1,0],[1,1],[5,1], [34,4]]
	testObj = Matrix(data2)

	ret = UML.run(toCall("GaussianNB"), trainingObj, testX=testObj, trainY=trainingYObj)


def testSciKitLearnUndiagnosed():
	""" Test scikitLearn on learners with previously undiagnosed crashes """
	variables = ["x1","x2"]
	data = [[1,0], [3,3], [50,0]]
	trainingObj = Matrix(data,variables)
	dataY = [[0],[1],[2]]
	trainingYObj = Matrix(dataY)

	data2 = [[1,0],[1,1],[5,1], [34,4]]
	testObj = Matrix(data2)

	ret = UML.run(toCall("EllipticEnvelope"), trainingObj, testX=testObj)
	#requires Y data
	ret = UML.run(toCall("MultinomialNB"), trainingObj, testX=testObj, trainY=trainingYObj)
	ret = UML.run(toCall("CCA"), trainingObj, testX=testObj, trainY=trainingYObj)
	ret = UML.run(toCall("PLSCanonical"), trainingObj, testX=testObj, trainY=trainingYObj)
	ret = UML.run(toCall("IsotonicRegression"), trainingObj, testX=testObj)



def testSciKitLearnListLearners():
	""" Test scikit learn's listSciKitLearnLearners() by checking the output for those learners we unit test """

	ret = UML.listLearners(packageName)

	assert 'KMeans' in ret
	assert 'LinearRegression' in ret

	toExclude = ['GaussianNB']

	for name in ret:
		if name not in toExclude:
			print name
			params = UML.learnerParameters(toCall(name))
			assert params is not None
			defaults = UML.learnerDefaultValues(toCall(name))
			for pSet in params:
				for dSet in defaults:
					for key in dSet.keys():
						assert key in pSet

	



