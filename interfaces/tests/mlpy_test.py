"""
Unit tests for mlpy_interface.py

"""

import UML

import numpy.testing

from UML.interfaces.mlpy_interface_old import setMlpyLocation
from UML.interfaces.mlpy_interface_old import getMlpyLocation
from test_helpers import checkLabelOrderingAndScoreAssociations
from UML.data import Matrix

def testMlpyLocation():
	""" Test setMlpyLocation() """
	path = '/test/path/mlpy'
	setMlpyLocation(path)

	assert getMlpyLocation() == path


def testMlpyHandmadeSVMClassification():
	""" Test mlpy() by calling on SVM classification with handmade output """

	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2], [3,1,500]]
	trainingObj = Matrix(data,variables)

	data2 = [[2,3],[-200,0]]
	testObj = Matrix(data2)

	ret = UML.run("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={})

	assert ret is not None

	expected = [[1.]]
	expectedObj = Matrix(expected)

	numpy.testing.assert_approx_equal(ret.data[0,0],1.)
	

def testMlpyHandmadeLogisticRegression():
	""" Test mlpy() by calling on logistic regression on handmade output """

	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2], [3,1,500]]
	trainingObj = Matrix(data,variables)

	data2 = [[2,3],[-200,0]]
	testObj = Matrix(data2)

	ret = UML.run("mlpy.LDAC", trainingObj, trainY="Y", testX=testObj, output=None, arguments={"solver_type":"l2r_lr"})

	assert ret is not None

	expected = [[1.]]
	expectedObj = Matrix(expected)

	numpy.testing.assert_approx_equal(ret.data[0,0],1.)
	

def testMlpyHandmadeKNN():
	""" Test mlpy() by calling on knn classification on handmade output """

	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2]]
	trainingObj = Matrix(data,variables)

	data2 = [[2,3],[0,0]]
	testObj = Matrix(data2)

	ret = UML.run("mlpy.KNN", trainingObj, trainY="Y", testX=testObj, output=None, arguments={"k":1})

	assert ret is not None

	numpy.testing.assert_approx_equal(ret.data[0,0],1.)
	numpy.testing.assert_approx_equal(ret.data[1,0],0.)

def testMlpyHandmadePCA():
	""" Test mlpy() by calling PCA and checking the output has the correct dimension """
	data = [[1,1,1], [2,2,2], [4,4,4]]
	trainingObj = Matrix(data)

	data2 = [[4,4,4]]
	testObj = Matrix(data2)

	ret = UML.run("mlpy.PCA", trainingObj, testX=testObj, output=None, arguments={'k':1})

	assert ret is not None
	# check return has the right dimension
	assert len(ret.data[0]) == 1 


def testMlpyHandmadeKernelPCA():
	""" Test mlpy() by calling PCA with a kernel transformation, checking the output has the correct dimension """
	data = [[1,1], [2,2], [3,3]]
	trainObj = Matrix(data)

	data2 = [[4,4]]
	testObj = Matrix(data2)

	ret = UML.run("mlpy.KPCA", trainObj, testX=testObj, output=None, arguments={"kernel":"KernelGaussian", 'k':1})

	assert ret is not None
	# check return has the right dimension
	assert len(ret.data[0]) == 1


def testMlpyScoreMode():
	""" Test mlpy() scoreMode flags"""
	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2]]
	trainingObj = Matrix(data,variables)

	data2 = [[2,3],[-200,0]]
	testObj = Matrix(data2)

	# default scoreMode is 'label'
	ret = UML.run("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	bestScores = UML.run("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert bestScores.pointCount == 2
	assert bestScores.featureCount == 2

	allScores = UML.run("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
	assert allScores.pointCount == 2
	assert allScores.featureCount == 3

	checkLabelOrderingAndScoreAssociations([0,1,2], bestScores, allScores)

def testMlpyScoreModeBinary():
	""" Test mlpy() scoreMode flags, binary case"""
	variables = ["Y","x1","x2"]
	data = [[1,1,1], [1,0,1],[1,-1,-1], [-1,30,2], [-1,30,3], [-1,34,4]]
	trainingObj = Matrix(data,variables)

	data2 = [[2,1],[25,0]]
	testObj = Matrix(data2)

	# default scoreMode is 'label'
	ret = UML.run("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	bestScores = UML.run("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert bestScores.pointCount == 2
	assert bestScores.featureCount == 2

	allScores = UML.run("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
	assert allScores.pointCount == 2
	assert allScores.featureCount == 2

	checkLabelOrderingAndScoreAssociations([-1,1], bestScores, allScores)


def testMlpyListLearners():
	""" Test mlpy's listMlpyLearners() by checking the output for those learners we unit test """

	ret = UML.listLearners('mlpy')

	assert 'KPCA' in ret
	assert 'PCA' in ret
	assert 'KNN' in ret
	assert "LibLinear" in ret
	assert "LibSvm" in ret

	toExclude = ['ClassTree', 'KNN', 'KernelAdatron', 'LibLinear', 'LibSvm', 'MaximumLikelihoodC']

	for name in ret:
		if name not in toExclude:
			params = UML.learnerParameters('mlpy.' + name)
			assert params is not None
			defaults = UML.learnerDefaultValues('mlpy.' + name)
			for key in defaults.keys():
				assert key in params

