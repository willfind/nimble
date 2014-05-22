"""
Unit tests for scikit_learn_interface.py

"""

import numpy.testing
import scipy.sparse
from numpy.random import rand, randint

import UML

from UML.customLearners import RidgeRegression
from UML.customLearners import KNNClassifier
from UML.interfaces.tests.test_helpers import checkLabelOrderingAndScoreAssociations
from UML.data import Matrix
from UML.data import Sparse

from UML.umlHelpers import generateClusteredPoints

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

	ret = UML.trainAndApply(toCall("LinearRegression"), trainingObj, trainY="Y", testX=testObj, output=None, arguments={})

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

	ret = UML.trainAndApply(toCall('SGDRegressor'), trainX=obj, trainY=cols[0], testX=testObj)

	assert ret is not None

def testSciKitLearnHandmadeClustering():
	""" Test sciKitLearn() by calling a clustering classifier with known output """
	variables = ["x1","x2"]
	data = [[1,0], [3,3], [5,0],]
	trainingObj = Matrix(data,variables)

	data2 = [[1,0],[1,1],[5,1], [3,4]]
	testObj = Matrix(data2)

	ret = UML.trainAndApply(toCall("KMeans"), trainingObj, testX=testObj, output=None, arguments={'n_clusters':3})

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

	ret = UML.trainAndApply(toCall('MiniBatchKMeans'), trainData, trainY=2, testX=testData, arguments={'n_clusters':2})
	
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
	ret = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	bestScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert bestScores.pointCount == 2
	assert bestScores.featureCount == 2

	allScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
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
	ret = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	bestScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert bestScores.pointCount == 2
	assert bestScores.featureCount == 2

	allScores = UML.trainAndApply(toCall("SVC"), trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
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

	UML.trainAndApply(toCall("GMM"), trainingObj, testX=testObj, arguments={'n_components':3})
	UML.trainAndApply(toCall("DPGMM"), trainingObj, testX=testObj)
	UML.trainAndApply(toCall("VBGMM"), trainingObj, testX=testObj)


#def testSciKitLearnObsAsArgumentName():
#	""" Test scikitLearn() by calling learners with 'obs' instead of 'X' as a fit/predict argument """
#	data = [[1,3,], [2,3], [0,1], [3,0], [3,2]]
#	trainingObj = Matrix(data)

#	data2 = [[2,3],[1,2],[0,1], [1,3]]
#	testObj = Matrix(data2)

#	ret = UML.trainAndApply(toCall("GMMHMM"), trainingObj, testX=testObj, arguments={'n_components':3})
#	ret = UML.trainAndApply(toCall("GaussianHMM"), trainingObj, testX=testObj)
#	ret = UML.trainAndApply(toCall("MultinomialHMM"), trainingObj, testX=testObj)

def testSciKitLearnArgspecFailures():
	""" Test scikitLearn() on those learners that cannot be passed to inspect.getargspec """
	variables = ["x1","x2"]
	data = [[1,0], [3,3], [50,0]]
	trainingObj = Matrix(data,variables)

	dataY = [[0],[1],[2]]
	trainingYObj = Matrix(dataY)

	data2 = [[1,0],[1,1],[5,1], [34,4]]
	testObj = Matrix(data2)

	UML.trainAndApply(toCall("GaussianNB"), trainingObj, testX=testObj, trainY=trainingYObj)
	# data dependent?
	UML.trainAndApply(toCall("MultinomialNB"), trainingObj, testX=testObj, trainY=trainingYObj)

def testSciKitLearnCrossDecomp():
	""" Test SKL on learners which take 2d Y data """
	variables = ["x1","x2"]
	data = [[1,0], [3,3], [50,0], [12, 3], [8, 228]]
	trainingObj = Matrix(data,variables)
	dataY = [[0,1],[0,1],[2,2],[1,30], [5,21]]
	trainingYObj = Matrix(dataY)

	data2 = [[1,0],[1,1],[5,1], [34,4]]
	testObj = Matrix(data2)

	UML.trainAndApply(toCall("CCA"), trainingObj, testX=testObj, trainY=trainingYObj)
	UML.trainAndApply(toCall("PLSCanonical"), trainingObj, testX=testObj, trainY=trainingYObj)
	UML.trainAndApply(toCall("PLSRegression"), trainingObj, testX=testObj, trainY=trainingYObj)
	UML.trainAndApply(toCall("PLSSVD"), trainingObj, testX=testObj, trainY=trainingYObj)


def testSciKitLearnListLearners():
	""" Test scikit learn's listSciKitLearnLearners() by checking the output for those learners we unit test """

	ret = UML.listLearners(packageName)

	assert 'KMeans' in ret
	assert 'LinearRegression' in ret

	toExclude = []

	for name in ret:
		if name not in toExclude:
			params = UML.learnerParameters(toCall(name))
			assert params is not None
			defaults = UML.learnerDefaultValues(toCall(name))
			for pSet in params:
				for dSet in defaults:
					for key in dSet.keys():
						assert key in pSet


def testCustomRidgeRegressionCompare():
	""" Sanity check for custom RidgeRegression, compare results to SKL's Ridge """
	data = [[0,1,2], [13,12,4], [345,233,76]]
	trainObj = Matrix(data)

	data2 = [[122,34],[76,-3]]
	testObj = Matrix(data2)

	UML.registerCustomLearner('Custom', RidgeRegression)

	name = 'Custom.RidgeRegression'
	ret1 = UML.trainAndApply(name, trainX=trainObj, trainY=0, testX=testObj, arguments={'lamb':1})
	ret2 = UML.trainAndApply("Scikitlearn.Ridge", trainX=trainObj, trainY=0, testX=testObj, arguments={'alpha':1, 'fit_intercept':False})
	

	assert ret1.isApproximatelyEqual(ret2)

def testCustomRidgeRegressionCompareRandomized():
	""" Sanity check for custom RidgeRegression, compare results to SKL's Ridge on random data"""
	trainObj = UML.createRandomData("Matrix", 1000, 60, .1)
	testObj = UML.createRandomData("Matrix", 100, 59, .1)

	UML.registerCustomLearner('Custom', RidgeRegression)

	name = 'Custom.RidgeRegression'
	ret1 = UML.trainAndApply(name, trainX=trainObj, trainY=0, testX=testObj, arguments={'lamb':1})
	ret2 = UML.trainAndApply("Scikitlearn.Ridge", trainX=trainObj, trainY=0, testX=testObj, arguments={'alpha':1, 'fit_intercept':False})
	
	assert ret1.isApproximatelyEqual(ret2)


def testCustomKNNClassficationCompareRandomized():
	""" Sanity check on custom KNNClassifier, compare to SKL's KNeighborsClassifier on random data"""
	trainX, ignore, trainY = generateClusteredPoints(5, 10, 5, addFeatureNoise=True, addLabelNoise=False, addLabelColumn=False)
	testX, ignore, testY = generateClusteredPoints(5, 5, 5, addFeatureNoise=True, addLabelNoise=False, addLabelColumn=False)

	UML.registerCustomLearner('Custom', KNNClassifier)

	cusname = 'Custom.KNNClassifier'
	sklname = "Scikitlearn.KNeighborsClassifier"
	ret1 = UML.trainAndApply(cusname, trainX, trainY=trainY, testX=testX, k=5)
	ret2 = UML.trainAndApply(sklname, trainX, trainY=trainY, testX=testX, n_neighbors=5, algorithm='brute')
	
	assert ret1.isApproximatelyEqual(ret2)
