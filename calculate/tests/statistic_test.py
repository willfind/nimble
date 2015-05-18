
# Most of the functions in UML.calculate.statitic are tested not directly
# in this module, but through the functions that call them: featureReport
# in UML.logger.tests.data_set_analyzier_tests and in the data
# hierarchy in UML.data.tests.query_backend

import numpy as np

from nose.tools import assert_almost_equal

from UML import createData
from UML.calculate import standardDeviation
from UML.calculate import quartiles

def testStDev():
    dataArr = np.array([[1], [1], [3], [4], [2], [6], [12], [0]])
    testRowList = createData('List', data=dataArr, featureNames=['nums'])
    stDevContainer = testRowList.applyToFeatures(standardDeviation, inPlace=False)
    stDev = stDevContainer.copyAs(format="python list")[0][0]
    assert_almost_equal(stDev, 3.6379, 3)

def testQuartilesSingleton():
    raw = [5]
    obj = createData("Matrix", raw)
    ret = quartiles(obj.pointView(0))

    assert ret == (5,5,5)

def testQuartilesEvens():
    raw = [1,1,3,3]
    ret = quartiles(raw)
    assert ret == (1,2,3)
    
    raw = [1,1,7,3,5,5,7,3]
    ret = quartiles(raw)
    assert ret == (2,4,6)

    raw = [7, 15, 36, 39, 40, 41]
    ret = quartiles(raw)
    assert ret == (15, 37.5, 40)

def testQuartilesPlusOne():
    raw = [1,3,2,2,2]
    ret = quartiles(raw)
    assert ret == (1.75, 2, 2.25)
    
    raw = [1,1,3,3,5,7,7,9,9]
    ret = quartiles(raw)
    assert ret == (2.5, 5, 7.5)

def testQuartilesPlusThree():
    raw = [1,1,2,4,3,5,5]
    # sorted: [1,1,2,3,4,5,5]
    ret = quartiles(raw)
    assert ret == (1.25, 3, 4.75)
    
    raw = [5,1,1,2,2,3,4,4,5,5,1]
    # sorted: [1,1,1,2,2,3,4,4,5,5,5]
    ret = quartiles(raw)
    assert ret == (1.25,3,4.75)

    raw = [6, 7, 15, 36, 39, 40, 41, 42, 43, 47, 49]
    ret = quartiles(raw)
    assert ret == (20.25, 40, 42.75)
