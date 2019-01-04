from __future__ import absolute_import
import UML
from UML.customLearners.multioutput_linear_regression import MultiOutputLinearRegression
from UML.configuration import configSafetyWrapper

# test for failure to import?

@configSafetyWrapper
def test_MultiOutputLinearRegression_simple():
    data = [[0, 1, 0], [0, 0, 2], [12, 0, 0], [2, 2, 2]]
    trainX = UML.createData('Matrix', data)

    data = [[10, -10], [2, -2], [1200, -1200], [222, -222]]
    trainY = UML.createData('Matrix', data)

    trainY0 = trainY.features.copy(0)
    trainY1 = trainY.features.copy(1)

    data = [[5, 5, 5], [0, 0, 1]]
    testX = UML.createData('Matrix', data)

    UML.registerCustomLearner('custom', MultiOutputLinearRegression)

    name = 'Custom.MultiOutputLinearRegression'
    retMulti = UML.trainAndApply(name, trainX=trainX, trainY=trainY, testX=testX)

    UML.deregisterCustomLearner('custom', 'MultiOutputLinearRegression')

    name = 'scikitlearn.LinearRegression'
    ret0 = UML.trainAndApply(name, trainX=trainX, trainY=trainY0, testX=testX)
    ret1 = UML.trainAndApply(name, trainX=trainX, trainY=trainY1, testX=testX)

    assert retMulti[0, 0] == ret0[0]
    assert retMulti[0, 1] == ret1[0]
    assert retMulti[1, 0] == ret0[1]
    assert retMulti[1, 1] == ret1[1]
