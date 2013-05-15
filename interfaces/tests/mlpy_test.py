"""
Unit tests for mlpy_interface.py

"""

import numpy.testing

from ..mlpy_interface import *
from ...processing.dense_matrix_data import DenseMatrixData as DMData
#from ...processing.row_list_data import RowListData as RLData

def testMlpyLocation():
	""" Test setMlpyLocation() """
	path = '/test/path/mlpy'
	setMlpyLocation(path)

	assert getMlpyLocation() == path


def testMlpyHandmadeSVMClassification():
	""" Test mlpy() by calling on SVM classification with handmade output """

	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2], [3,1,500]]
	trainingObj = DMData(data,variables)

	data2 = [[2,3],[-200,0]]
	testObj = DMData(data2)

	ret = mlpy("LibSvm", trainingObj, testObj, dependentVar="Y", arguments={})

	assert ret is not None

	expected = [[1.]]
	expectedObj = DMData(expected)

	numpy.testing.assert_approx_equal(ret.data[0,0],1.)
	

def testMlpyHandmadeLogisticRegression():
	""" Test mlpy() by calling on logistic regression on handmade output """

	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2], [3,1,500]]
	trainingObj = DMData(data,variables)

	data2 = [[2,3],[-200,0]]
	testObj = DMData(data2)

	ret = mlpy("LibLinear", trainingObj, testObj, output=None, dependentVar="Y", arguments={"solver_type":"l2r_lr"})

	assert ret is not None

	expected = [[1.]]
	expectedObj = DMData(expected)

	numpy.testing.assert_approx_equal(ret.data[0,0],1.)
	

def testMlpyHandmadeKNN():
	""" Test mlpy() by calling on knn classification on handmade output """

	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2]]
	trainingObj = DMData(data,variables)

	data2 = [[2,3],[0,0]]
	testObj = DMData(data2)

	ret = mlpy("KNN", trainingObj, testObj, output=None, dependentVar="Y", arguments={"k":1})

	assert ret is not None

	numpy.testing.assert_approx_equal(ret.data[0,0],1.)
	numpy.testing.assert_approx_equal(ret.data[1,0],0.)

def testMlpyHandmadePCA():
	""" Test mlpy() by calling PCA and checking the output has the correct dimension """
	data = [[1,1,1], [2,2,2], [4,4,4]]
	trainingObj = DMData(data)

	data2 = [[4,4,4]]
	testObj = DMData(data2)

	ret = mlpy("PCA", trainingObj, testObj, output=None, arguments={'k':1})

	assert ret is not None
	# check return has the right dimension
	assert len(ret.data[0]) == 1 


def testMlpyHandmadeKernelPCA():
	""" Test mlpy() by calling PCA with a kernel transformation, checking the output has the correct dimension """
	data = [[1,1], [2,2], [3,3]]
	trainObj = DMData(data)

	data2 = [[4,4]]
	testObj = DMData(data2)

	ret = mlpy("KPCA", trainObj, testObj, output=None, arguments={"kernel":"KernelGaussian", 'k':1})

	assert ret is not None
	# check return has the right dimension
	assert len(ret.data[0]) == 1


def testMlpyScoreMode():
	""" Test mlpy() returns the right dimensions of output when given different scoreMode flags"""
	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2]]
	trainingObj = DMData(data,variables)

	data2 = [[2,3],[-200,0]]
	testObj = DMData(data2)

	# default scoreMode is 'label'
	ret = mlpy("LibSvm", trainingObj, testObj, dependentVar="Y", arguments={})
	assert ret.points() == 2
	assert ret.features() == 1

	ret = mlpy("LibSvm", trainingObj, testObj, dependentVar="Y", arguments={}, scoreMode='bestScore')
	assert ret.points() == 2
	assert ret.features() == 2

	ret = mlpy("LibSvm", trainingObj, testObj, dependentVar="Y", arguments={}, scoreMode='allScores')
	assert ret.points() == 2
	assert ret.features() == 3


def testMlpyListAlgorithms():
	""" Test mlpy's listAlgorithms() by checking the output for those algorithms we unit test """

	ret = listAlgorithms()

	assert 'KPCA' in ret
	assert 'PCA' in ret
	assert 'KNN' in ret
	assert "LibLinear" in ret
	assert "LibSvm" in ret




