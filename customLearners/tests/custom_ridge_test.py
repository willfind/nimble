
import UML

from nose.tools import *
import numpy.testing

from UML.data import Matrix
from UML.customLearners.ridge_regression import RidgeRegression


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

