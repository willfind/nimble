
import UML

from nose.tools import *
import numpy.testing

from UML.exceptions import ArgumentException
from UML.data import Matrix
from UML.interfaces.ridge_regression import RidgeRegression


def testRidgeRegressionShapes():
	""" Test ridge regression by checking the shapes of the inputs and outputs """

	data = [[0,0,0], [4,3,1], [12,15,-3], ]
	trainObj = Matrix(data)

	data2 = [[5.5,5],[20,-3]]
	testObj = Matrix(data2)

	UML.registerCustomLearner('Custom', RidgeRegression)

	name = 'Custom.RidgeRegression'
	ret = UML.trainAndApply(name, trainX=trainObj, trainY=0, testX=testObj, arguments={'lamb':0})

	assert ret.pointCount == 2
	assert ret.featureCount == 1
	numpy.testing.assert_approx_equal(ret[0,0], 10.5, significant=3)
	numpy.testing.assert_approx_equal(ret[1,0], 18, significant=2)


def testRidgeRegressionCompare():
	data = [[0,1,2], [13,12,4], [345,233,76]]
	trainObj = Matrix(data)

	data2 = [[122,34],[76,-3]]
	testObj = Matrix(data2)

	UML.registerCustomLearner('Custom', RidgeRegression)

	name = 'Custom.RidgeRegression'
	ret1 = UML.trainAndApply(name, trainX=trainObj, trainY=0, testX=testObj, arguments={'lamb':0})
	ret2 = UML.trainAndApply("Scikitlearn.Ridge", trainX=trainObj, trainY=0, testX=testObj, arguments={'alpha':0, 'fit_intercept':False})
	
	assert ret1.isApproximatelyEqual(ret2)

